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

# import methods from other files
import input_eeg_float_subj
from input_eeg_float_subj import input_pipeline
from eeg_cnn_preictal import encoder_template, decoder_template, inference_template, loss, training, evaluation, evaldataset, CLASS_LOSS, RECON_LOSS

NTIMEPOINTS = input_eeg_float_subj.NTIMEPOINTS
NCHANNELS = input_eeg_float_subj.NCHANNELS

### MAIN: runs the functions defined in eeg_autoencoder_supervised for detection of preictal periods

# general training parameters
train_folder = "C:\\Users\\CASHLAB_RECON\\Documents\\tf\\preictal 512\\train"
test_folder = "C:\\Users\\CASHLAB_RECON\\Documents\\tf\\preictal 512\\test"
# train_folder = "C:\\Users\\cashlab\\Documents\\tf\\preictal dataset shuffle\\train"
# test_folder = "C:\\Users\\cashlab\\Documents\\tf\\preictal dataset shuffle\\test"
filenames_train = [join(train_folder, f) for f in listdir(train_folder) if isfile(join(train_folder, f))]
filenames_test = [join(test_folder, f) for f in listdir(test_folder) if isfile(join(test_folder, f))]
global_step = tf.Variable(0, trainable=False)
num_samp_train = 671665
num_samp_test = 164474
batch_size = 512
num_epochs = 60
it_per_epoch = math.floor(num_samp_train/batch_size)

# layers whose weights should be saved to binary files (and binary files names) for each epoch
# list of tuples with syntax: (tensor name, savefile name)
weights_save = [("encoder/W_conv1:0", "W1.bin")]

# directory for keeping summary files
logdir = "C:\\preictal"
logfiles_old = [join(logdir, f) for f in listdir(logdir) if isfile(join(logdir, f))]
for f in logfiles_old: os.remove(f)

# import pipelines on CPU
with tf.device('/cpu:0'):
    # train input
    with tf.name_scope('train_input'):
        image_batch, label_batch, animal_id_batch, time_sz_batch = input_pipeline(filenames_train, batch_size, bShuffle=True)
    # test input
    with tf.name_scope('test_input'):
        image_batch_test, label_batch_test, animal_id_batch_test, time_sz_batch_test = input_pipeline(filenames_test, batch_size, bShuffle=False)

# assigment for future functions
x = tf.reshape(image_batch, [batch_size, NCHANNELS, NTIMEPOINTS, 1])
y_ = label_batch
xtest = tf.reshape(image_batch_test, [batch_size, NCHANNELS, NTIMEPOINTS, 1])
y_test = label_batch_test

# inference and encoding ops (template so it can be shared between train and test)
# templates for each function
encoder = tf.make_template('encoder', encoder_template)
decoder = tf.make_template('decoder', decoder_template)
inference = tf.make_template('inference', inference_template)

# encoding ops
code = encoder(x,batch_size, bEval = False)
with tf.name_scope('encoder_eval_train'):
    code_eval_train = encoder(x,batch_size, bEval = True)
with tf.name_scope('encoder_eval_test'):
    code_eval_test = encoder(xtest,batch_size, bEval = True)

# decoding op (if autoencoder is trained)
if RECON_LOSS>0:
    x_logits = decoder(code,batch_size, bEval = False)
    x_reconstructed = x_logits
else:
    x_reconstructed = x

# inference ops (if supervised training)
if CLASS_LOSS>0:
    y = inference(code, False)
    with tf.name_scope('inference_eval_train'):
        yeval = inference(code_eval_train, True)
    with tf.name_scope('inference_eval_test'):
        yeval_test = inference(code_eval_test, True)

    # accuracy for single batch
    with tf.name_scope('accuracy'):
        accuracy = evaluation(y, y_)
    tf.summary.scalar('Mini-batch accuracy', accuracy)
    with tf.name_scope('accuracy_eval_train'):
        accuracy_eval = evaluation(yeval, y_)
    with tf.name_scope('accuracy_eval_test'):
        accuracy_eval_test = evaluation(yeval_test, y_test)
else:
    y = y_

# compute loss op
lossval = loss(x_reconstructed, x, y,y_)
tf.summary.scalar('Mini-batch loss', lossval)

# training op
train_step = training(lossval, global_step)

# variables initializer
init = tf.global_variables_initializer()

# merged summary for easy writing
merged_summary = tf.summary.merge_all()

# op to save the variables to file
train_vars = tf.trainable_variables()
save_dict = dict()
for v in train_vars:
    save_dict.update({v.name : v})
saver = tf.train.Saver(save_dict, max_to_keep=15)


# training step in loop
# with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
with tf.Session() as sess:
    # initialize variables
    sess.run(init)
    # initialize summary writer for this session
    writer = tf.summary.FileWriter(logdir, sess.graph)

    # for each element in the list of weights tensors to save, open bin file and write
    # the dimensions of the tensor
    files_weights = []
    tensors_weights = []
    for w in weights_save:
        f = open(join(logdir, w[1]), 'wb')
        t = sess.graph.get_tensor_by_name(w[0])
        t_np = sess.run(t)
        f.write(np.asarray(t_np.shape).astype('float32').tobytes())
        files_weights.append(f)
        tensors_weights.append(t)
        print("Tensor : " + w[0] + " with shape " + str(t_np.shape) + " will be saved to " +  w[1])

    # finalize the graph and start timing
    start_time = time.time()
    test_time = start_time
    sess.graph.finalize()


    for it in range(1,num_epochs*it_per_epoch+1):
        # train step
        #_, lossv, acc_batch, l, merged = sess.run([train_step, lossval, accuracy, label_batch[0], merged_summary])
        _, loss, merged = sess.run([train_step, lossval, merged_summary])
        # print(t)
        writer.add_summary(merged, it)
        if it % 100 == 0:
            print(time.time()-test_time)
            test_time = time.time()

        if it == 1:
            epoch_start = time.time()
        if it % it_per_epoch == 0:
            print(it)
            #print(l)
            if it == 0:
                print("Initialization")
            else:
                print("Epoch {} ".format(int(it/it_per_epoch)))
                print("\tDuration[s]: {}".format(time.time()-epoch_start))
                print("\tSamples per second: {}".format(batch_size*it_per_epoch/(time.time()-epoch_start)))
                # evaluate model on train and test (valid) datasets
                acc = evaldataset(accuracy_eval, sess, num_samp_train, batch_size)
                print("\tTraining accuracy: {}".format(acc))
                test_acc = evaldataset(accuracy_eval_test, sess, num_samp_test, batch_size)
                print("\tTest accuracy: {}".format(test_acc))
                # write them to the summary
                summary = tf.Summary()
                summary.value.add(tag='Train accuracy', simple_value=acc)
                summary.value.add(tag='Test accuracy', simple_value=test_acc)
                writer.add_summary(summary, it)
                # save the model
                save_path = saver.save(sess, join(logdir, 'model_backup.ckpt'), global_step=it)
                epoch_start = time.time()
                # if specified, keep a copy of weights for each epoch
                for i in range(len(tensors_weights)):
                    w = sess.run(tensors_weights[i])
                    files_weights[i].write(w.astype('float32').tobytes())


    elapsed_time = time.time() - start_time
    print(elapsed_time)


    for f in files_weights:
        f.close()

    # write the weights to a binary file
    train_vars = tf.trainable_variables()
    # the weights and bias are the trainable variables for the session
    net_vars = sess.run(train_vars)
    with open(join(logdir, 'model_backup.bin'), 'wb') as f:
        for var in net_vars: f.write(var.astype('float32').tobytes())
