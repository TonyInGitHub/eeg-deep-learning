import tensorflow as tf
import math
import input_eeg_float_subj
import numpy as np


NTIMEPOINTS= input_eeg_float_subj.NTIMEPOINTS
NCHANNELS = input_eeg_float_subj.NCHANNELS
NCLASSES = input_eeg_float_subj.NCLASSES

DECAY_LEARNING_RATE = False
INITIAL_LEARNING_RATE = 0.0005
DECAY_STEPS = 2000
LEARNING_RATE_DECAY_FACTOR = 0.1
L2_REG = 0 # 0.01
CLASS_LOSS = 1
RECON_LOSS = 5 #5

std = 0.1

### INFERENCE
def encoder_template(x, batch_size, bEval):

    if bEval:
        l2_reg = 0
        bSummary = False
    else:
        l2_reg = L2_REG
        bSummary = True
        print('ENCODER')
        print("Input: [" + str(x.shape) + "]")

    # log transform
    x = tf.sign(x)*tf.log(tf.abs(x)+1)

    # x: 3 x 2560 x 1
    y1 = convLayer(x, strides = [1,1,10,1], kernel = [1,52,1,10], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv1') # 3 x 256 x 10
    yp1 = maxpoolLayer(y1, strides = [1,1,4,1], kernel = [1,1,4,1], bSummary = bSummary, name = 'pool1') # 3 x 64 x 10
    y2 = convLayer(yp1, strides = [1,1,4,1], kernel = [1,8,10,20], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv2') # 3 x 16 x 20
    yp2 = maxpoolLayer(y2, strides = [1,1,2,1], kernel = [1,1,2,1], bSummary = bSummary, name = 'pool2') # 3 x 8 x 20
    y3 = convLayer(yp2, strides = [1,1,1,1], kernel = [3,8,20,20], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv3') # 3 x 8 x 20
    y4 = convLayer(y3, strides = [1,1,1,1], kernel = [3,8,20,30], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv4') # 3 x 8 x 30
    y5 = convLayer(y4, strides = [1,1,1,1], kernel = [3,8,30,30], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv5') # 3 x 8 x 30
    y6 = convLayer(y5, strides = [1,1,1,1], kernel = [3,8,30,30], wd=l2_reg, std=std, activation='relu', bSummary=bSummary, name='conv6') # 3 x 8 x 30

    return y6


def decoder_template(x, batch_size, bEval): # 3 x 8 x 20

    if bEval or RECON_LOSS==0:
        l2_reg = 0
        bSummary = False
    else:
        l2_reg = L2_REG
        bSummary = True
        print('DECODER')

    y1 = deconvLayer(x, strides = [1,1,1,1], kernel = [3,8,30,30], output_shape = [batch_size, 3, 8, 30], wd=l2_reg, std=std, activation='tanh', bSummary=bSummary, name='deconv1')
    y2 = deconvLayer(y1, strides = [1,1,1,1], kernel = [3,8,30,30], output_shape = [batch_size, 3, 8, 30], wd=l2_reg, std=std, activation='tanh', bSummary=bSummary, name='deconv2')
    y3 = deconvLayer(y2, strides = [1,1,1,1], kernel = [3,8,20,30], output_shape = [batch_size, 3, 8, 20], wd=l2_reg, std=std, activation='tanh', bSummary=bSummary, name='deconv3')
    y4 = deconvLayer(y3, strides = [1,1,1,1], kernel = [3,8,20,20], output_shape = [batch_size, 3, 8, 20], wd=l2_reg, std=std, activation='tanh', bSummary=bSummary, name='deconv4')
    yu1 = upsampleLayer(y4, size = [batch_size, 3, 16, 20], bSummary=bSummary, name='upsample1')
    y5 = deconvLayer(yu1, strides = [1,1,4,1], kernel = [1,8,10,20], output_shape = [batch_size, 3, 64, 10], wd=l2_reg, std=std, activation='tanh', bSummary=bSummary, name='deconv5')
    yu2 = upsampleLayer(y5, size = [batch_size, 3, 256, 10], bSummary=bSummary, name='upsample2')
    y6 = deconvLayer(yu2, strides = [1,1,10,1], kernel = [1,52,1,10], output_shape = [batch_size, 3, 2560, 1], wd=l2_reg, std=std, activation='none', bSummary=bSummary, name='deconv6', useBias=False)

    # convert back from log form
    y6 = tf.sign(y6)*(tf.exp(tf.abs(y6))-1)

    return y6

### INFERENCE
def inference_template(code, bEval):

    if bEval or RECON_LOSS==0:
        keep_prob = 1
        l2_reg = 0
        bSummary = False
    else:
        keep_prob = 0.5
        l2_reg = L2_REG
        bSummary = True

    flatsize = 3*8*30

    with tf.name_scope('fc1'):
        W_fc1 = variable_L2('W_fc1', shape = [flatsize,1024], initializer = tf.truncated_normal_initializer(stddev=std), wd = l2_reg, bSummary=bSummary)
        b_fc1 = variable_on_cpu('b_fc1', shape = [1024], initializer = tf.constant_initializer(std), bSummary=bSummary)
        # flatten output of previous layer to connec with fully connected
        h_conv5_flat = tf.reshape(code, [-1, flatsize])
        h_fc1 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc1) + b_fc1)

    with tf.name_scope('dropout'):
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    with tf.name_scope('fc2'):
        W_fc2 = variable_L2('W_fc2', shape = [1024, 2], initializer = tf.truncated_normal_initializer(stddev=std), wd = l2_reg, bSummary=bSummary)
        b_fc2 = variable_on_cpu('b_fc2', shape = [2], initializer = tf.constant_initializer(std), bSummary=bSummary)
        logits = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

        return logits

### LOSS
def loss(x_reconstructed, x, y, y_):

    # RECONSTRUCTION LOSS
    if RECON_LOSS > 0:
        with tf.name_scope('reconstruction_loss'):
            mse = RECON_LOSS * tf.reduce_mean(tf.pow(x - x_reconstructed, 2))
        tf.add_to_collection('losses', mse)
        tf.summary.scalar('Reconstruction loss', mse)

    # CLASSIFICATION LOSS
    if CLASS_LOSS > 0:
        with tf.name_scope('classification_loss'):
            class_weights = 1/(tf.reduce_sum(y_,axis = 0)+1e-6)
            # deduce weights for batch samples based on their true label
            weights = tf.reduce_sum(class_weights * y_, axis=1)
            # compute (unweighted) softmax cross entropy loss
            unweighted_losses = tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y)
            # apply the weights, relying on broadcasting of the multiplication
            weighted_losses = unweighted_losses * weights
            # reduce the result to get final loss
            cross_entropy = CLASS_LOSS * tf.reduce_sum(weighted_losses)/tf.cast(tf.count_nonzero(tf.reduce_sum(y_,axis=0)), tf.float32)
            tf.add_to_collection('losses', cross_entropy)
        tf.summary.scalar('Classification loss', cross_entropy)

    # get the total loss (loss+L2loss)
    total_loss = tf.add_n(tf.get_collection('losses'), name='total_loss')

    return total_loss

### TRAINING
def training(lossval, global_step):
    if DECAY_LEARNING_RATE:
        lr = tf.train.exponential_decay(INITIAL_LEARNING_RATE, global_step, DECAY_STEPS,
                                  LEARNING_RATE_DECAY_FACTOR, staircase=True)
    else:
        lr = INITIAL_LEARNING_RATE
    tf.summary.scalar('Learning rate', lr)
    #train_step = tf.train.RMSPropOptimizer(lr).minimize(lossval, global_step = global_step)

    train_step = tf.train.AdamOptimizer(lr).minimize(lossval, global_step = global_step)
    #train_step = tf.train.GradientDescentOptimizer(lr).minimize(lossval, global_step = global_step)
    #train_step = tf.train.MomentumOptimizer(lr, 0.9).minimize(lossval, global_step = global_step)

    return train_step

### EVALUATION
def evaluation(y, y_):
    class_weights = 1/(tf.reduce_sum(y_,axis = 0)+1e-6)
    # deduce weights for batch samples based on their true label
    weights = tf.reduce_sum(class_weights * y_, axis=1)
    # compute (unweighted) accuracy for each sample (0 or 1)
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    # apply the weights, relying on broadcasting of the multiplication
    weighted_acc = correct_prediction * weights
    # reduce the result to get final accuracy
    accuracy = tf.reduce_sum(weighted_acc)/tf.cast(tf.count_nonzero(tf.reduce_sum(y_,axis=0)), tf.float32)

    return accuracy

def evaldataset(accuracy_op, sess, num_samp, batch_size):
    num_it = math.floor(num_samp/batch_size)
    tot_accuracy = 0
    for _ in range(num_it):
        accuracy_temp = sess.run(accuracy_op)
        tot_accuracy += accuracy_temp/num_it

    return tot_accuracy


# HELPER FUNCTIONS
def convLayer(x, strides, kernel, wd, std, activation, bSummary, name, useBias=True):
    # initializer is hardcoded here
    # init = tf.keras.initializers.glorot_uniform()
    # init = tf.truncated_normal_initializer(stddev=std)
    init = tf.contrib.layers.xavier_initializer(uniform=True)
    with tf.name_scope(name):
        W = variable_L2('W_'+name, shape = kernel, initializer = init, wd=wd, bSummary=bSummary)
        b = variable_on_cpu('b_'+name, shape = kernel[-1], initializer = init, bSummary=bSummary)
        if activation=='relu':
            h = tf.nn.relu(conv2d(x, W, strides = strides) + b)
        elif activation=='tanh':
            h = tf.nn.tanh(conv2d(x, W, strides = strides) + b)
        else:
            h = conv2d(x, W, strides = strides) + b
    if bSummary: # display informaion in the command line and add the variable to the Tensorboard summary
        print('Convolutional layer created (' + name + ')')
        print('- Kernel: ' + str(kernel))
        print('- Strides: ' + str(strides))
        print('- Output size: ' + str(h.shape))
    return h

def deconvLayer(x, strides, kernel, output_shape, wd, std, activation, bSummary, name, useBias=True):
    # initializer is hardcoded here
    # init = tf.keras.initializers.glorot_uniform()
    # init = tf.truncated_normal_initializer(stddev=std)
    init = tf.contrib.layers.xavier_initializer(uniform=False)
    with tf.name_scope(name):
        if useBias:
            W = variable_L2('W_'+name, shape = kernel, initializer = init, wd=wd, bSummary=bSummary)
            b = variable_on_cpu('b_'+name, shape = kernel[-2], initializer = init, bSummary=bSummary)
            if activation=='relu':
                h = tf.nn.relu(deconv2d(x, W, output_shape=output_shape, strides = strides) + b)
            elif activation=='tanh':
                h = tf.nn.tanh(deconv2d(x, W, output_shape=output_shape, strides = strides) + b)
            else:
                h = deconv2d(x, W, output_shape=output_shape, strides = strides) + b
        else:
            W = variable_L2('W_'+name, shape = kernel, initializer = init, wd=wd, bSummary=bSummary)
            if activation=='relu':
                h = tf.nn.relu(deconv2d(x, W, output_shape=output_shape, strides = strides))
            elif activation=='tanh':
                h = tf.nn.tanh(deconv2d(x, W, output_shape=output_shape, strides = strides))
            else:
                h = deconv2d(x, W, output_shape=output_shape, strides = strides)
    if bSummary: # display informaion in the command line and add the variable to the Tensorboard summary
        print('Convolutional transpose layer created (' + name + ')')
        print('- Kernel: ' + str(kernel))
        print('- Strides: ' + str(strides))
        print('- Output size: ' + str(output_shape))
    return h


def maxpoolLayer(x, strides, kernel, bSummary, name):
    with tf.name_scope(name):
        h = max_pool(x, ksize = kernel, strides = strides)
    if bSummary: # display informaion in the command line and add the variable to the Tensorboard summary
        print('Max pooling layer created (' + name + ')')
        print('- Kernel: ' + str(kernel))
        print('- Strides: ' + str(strides))
        print('- Output size: ' + str(h.shape))
    return h

def upsampleLayer(x, size, bSummary, name):
    with tf.name_scope(name):
        h = tf.image.resize_images(x, size=(size[1], size[2]), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    if bSummary: # display informaion in the command line and add the variable to the Tensorboard summary
        print('Upsampling layer created (' + name + ')')
        print('- Output size: ' + str(h.shape))
    return h

def conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME'):
    return tf.nn.conv2d(x, W, strides=strides, padding=padding)

def deconv2d(x, W, output_shape, strides=[1, 1, 1, 1],  padding='SAME'):
    return tf.nn.conv2d_transpose(x, W, output_shape=output_shape, strides=strides, padding=padding)

def max_pool(x,ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1]):
    return tf.nn.max_pool(x, ksize, strides, padding='SAME')

def variable_on_cpu(name, shape, initializer, bSummary):
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
        if bSummary:
            tf.summary.histogram(name, var)
    return var

def variable_L2(name, shape, initializer, wd, bSummary):
    var = variable_on_cpu(name, shape, initializer, bSummary)
    if not(wd==0):
        weight_loss = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
        tf.add_to_collection('losses', weight_loss)
    return var
