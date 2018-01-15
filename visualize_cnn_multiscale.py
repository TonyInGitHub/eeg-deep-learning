from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import math
import tensorflow as tf
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import pylab
import time
from os import listdir
from os.path import isfile, join
import os.path
import os
import numpy as np

from eeg_cnn_preictal import encoder_template, decoder_template, inference_template, RECON_LOSS
import input_eeg_float_subj


NTIMEPOINTS = input_eeg_float_subj.NTIMEPOINTS
NCHANNELS = input_eeg_float_subj.NCHANNELS

def get_plot_index(lin, col, i):
    sub_i = math.floor(i/col)
    sub_j = i % col
    if sub_i > lin:
        sub_i = -1
    return sub_i, sub_j

def tffunc(*argtypes):
    # Helper that transforms TF-graph generating function into a regular one.
    placeholders = list(map(tf.placeholder, argtypes))
    def wrap(f):
        out = f(*placeholders)
        def wrapper(*args, **kw):
            return out.eval(dict(zip(placeholders, args)), session=kw.get('session'))
        return wrapper
    return wrap

# Helper function that uses TF to resize an image
def resize(img, size):
    #img = tf.expand_dims(img, 0)
    return tf.image.resize_bilinear(img, size)#[0,:,:,:]
resize = tffunc(np.float32, np.int32)(resize)


def calc_grad_tiled(img, t_grad, tile_size=100):
    # Compute the value of tensor t_grad over the image in a tiled way.
    #Random shifts are applied to the image to blur tile boundaries over
    # multiple iterations.
    # define the tiles only in the time domain of the EEG data
    sz = tile_size
    c, t = img.shape[:2]
    s = np.random.randint(sz, size=1)
    img_shift = np.roll(img, s, 1)
    grad = np.zeros_like(img)
    for y in range(0, max(t-sz//2, sz),sz):
        sub = img_shift[:,y:y+sz]
        g = sess.run(t_grad, {x:sub})
        grad[:,y:y+sz] = g
    return np.roll(grad, -s, 1)

def render_multiscale(t_obj, img0, iter_n=200, step=0.5, octave_n=3, octave_scale=1.4):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, x)[0] # behold the power of automatic differentiation!

    img = img0.copy()
    for octave in range(octave_n):
        if octave>0:
            sz = np.float32((img.shape[1], img.shape[2]*octave_scale))
            img = resize(img, np.int32(sz))
        for i in range(iter_n):
            g = calc_grad_tiled(img, t_grad)
            # normalizing the gradient, so the same step size should work
            g /= g.std()+1e-8         # for different layers and networks
            img += g*step

    return img

def render_naive(t_obj, img0, iter_n=200, step=0.5):
    t_score = tf.reduce_mean(t_obj) # defining the optimization objective
    t_grad = tf.gradients(t_score, x)[0] # behold the power of automatic differentiation!

    img = img0.copy()
    for i in range(iter_n):
        g, score = sess.run([t_grad, t_score], {x:img})
        # normalizing the gradient, so the same step size should work
        g /= g.std()+1e-8         # for different layers and networks
        img += g*step
    return np.transpose(np.squeeze(img))


logdir = "C:\\preictal"
id_torestore = -1 # index of the model to restore (-1 if last)
unitsInput = True
featuresDecoded = False
multiscale = False

batch_size = 1
x = tf.placeholder(tf.float32)#, shape=(1,NCHANNELS,NTIMEPOINTS))

# inference ops (template so it can be shared between train and test)
encoder = tf.make_template('encoder', encoder_template)
decoder = tf.make_template('decoder', decoder_template)
inference = tf.make_template('inference', inference_template)
code = encoder(x,batch_size, bEval = True)
y = inference(code, True)
if RECON_LOSS>0:
    x_logits = decoder(code,batch_size, bEval = False)
    x_reconstructed = x_logits

# restore variables from checkpoint
train_vars = tf.trainable_variables()
save_dict = dict()
for v in train_vars:
    save_dict.update({v.name : v})
print()
print('Variables to restore:')
for v in train_vars:
    print(v)
saver = tf.train.Saver(save_dict)
logdir_vis = "C:\\autoencoder_vis"
# get all files in logdir
logfiles = [join(logdir, f) for f in listdir(logdir) if isfile(join(logdir, f))]
# keep only the checkpoint files
logfiles = [l for l in logfiles if len(l.split('.'))>1]
ckptfiles = [l for l in logfiles if l.split('.')[-2][0:4]=="ckpt"]

# take the last (most recent) checkpoint / or the one specified
if id_torestore == -1:
    restorefile = ".".join(ckptfiles[-1].split('.')[:-1])
else:
    restorefiles = [c for c in ckptfiles if c.split('.')[-2][5:]==str(id_torestore)]
    restorefile = ".".join(restorefiles[-1].split('.')[:-1])
print()
print('Variables restored from:')
print(restorefile)
print()


with tf.Session() as sess:
    # saver.restore(sess, join(logdir, 'model_backup.ckpt-3360'))
    saver.restore(sess, restorefile)
    #out = sess.graph.get_tensor_by_name("encoder/encoder_conv2/Relu:0")
    #out = sess.graph.get_tensor_by_name("encoder/encoder_conv3/add:0")

    ## RECONSTRUCTION OF UNITS PREFERRED INPUT USING GRADIENT ASCENT OPTIMIZATION
    if unitsInput:
        # get the tensors that represent output from the layers
        tensor_names = ["encoder/conv1/add:0",
        "encoder/conv2/add:0",
        "encoder/conv3/add:0",
        "encoder/conv4/add:0",
        "encoder/conv5/add:0",
        "encoder/conv6/add:0",
        "inference/fc2/add:0"]#,
        # list of tensors that represent the kernel weights from the first layer
        weights_names = ["encoder/W_conv1:0"]


        tensor_list = []
        tensor_convn = []
        for n in tensor_names:
            tensor_list.append(sess.graph.get_tensor_by_name(n))
            tensor_convn.append(sess.graph.get_tensor_by_name(n).get_shape().as_list()[-1])
        print(tensor_list)
        print(tensor_convn)


        # get the kernel weights of the first layer and plot them
        for n in weights_names:
            #weights = train_vars[0]
            weights = sess.graph.get_tensor_by_name(n)
            nconv = weights.get_shape().as_list()[-1]
            col = math.floor(math.sqrt(14.0/9.0*nconv))
            lin = int(math.ceil(nconv/col))
            col = int(col)
            fig, axes = plt.subplots(lin, col)
            for n in range(nconv):
                i,j = get_plot_index(lin, col, n)
                axes[i,j].plot(np.squeeze(sess.run(weights[:,:,:,n])))

        # use derivation to find signal that has maximal activation for the layer outputs
        img_noise = np.random.uniform(size=(1,NCHANNELS,NTIMEPOINTS,1))*0.02
        for ind in range(len(tensor_convn)):
            nconv = tensor_convn[ind]
            col = math.floor(math.sqrt(14.0/9.0*nconv))
            lin = int(math.ceil(nconv/col))
            col = int(col)
            fig, axes = plt.subplots(lin, col)
            for n in range(nconv):
                i,j = get_plot_index(lin, col, n)
                if multiscale:
                    if len(tensor_list[ind].shape)==2:
                        img = render_multiscale(tensor_list[ind][:,n], img_noise)
                    else:
                        img = render_multiscale(tensor_list[ind][:,:,:,n], img_noise)
                    img = img.squeeze().transpose()
                else:
                    if len(tensor_list[ind].shape)==2:
                        img = render_naive(tensor_list[ind][:,n], img_noise)
                    else:
                        img = render_naive(tensor_list[ind][:,:,:,n], img_noise)
                if lin == 1 or col == 1:
                    axes[n].plot(img)
                    axes[n].set_xlim(2*NTIMEPOINTS/5, 3*NTIMEPOINTS/5)
                else:
                    axes[i,j].plot(img)
                    axes[i,j].set_xlim(2*NTIMEPOINTS/5, 3*NTIMEPOINTS/5)

    if featuresDecoded:
        # list of the indexes of the features to access
        feat_ind = [(0,1,0,29)]
        values_list = [-10,-0.5,0,0.5,10]
        # define code matrix
        code0 = np.zeros((1,3,1,30))
        for i in range(len(feat_ind)):
            fig, axes = plt.subplots(len(values_list), 1)
            for j in range(len(values_list)):
                code_np = code0
                code_np[feat_ind[i]] = values_list[j]
                x_reconstructed_np = sess.run(x_reconstructed, {code:code_np})
                img = x_reconstructed_np.squeeze().transpose()
                axes[j].plot(img)
    pylab.show()




















#
