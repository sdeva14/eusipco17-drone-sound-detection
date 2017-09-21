'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)
'''

from __future__ import print_function

import tensorflow as tf

from src_dd.dataset import *

import src_dd.features_drone as feat_dr
import src_dd.evaluation_drone as eval_dr
import src_dd.util_drone as util_dr
import src_dd.files as files_dr
import src_dd.modeling_manual_1 as modeling_manual
import src_dd.loadParam_manually as loadParam_manually

import os
import numpy
import csv
import warnings
import argparse
import textwrap
import math
import re
import time

import librosa
import scipy


def readDataset(params, files, dataset_evaluation_mode, num_folds):
    # print(params['general']['drone_dataset'])
    # print(params['path']['data'])

    dataset = eval(params['general']['drone_dataset'])(data_path=params['path']['data'], num_folds=num_folds);
    files = []
    for fold in dataset.folds(mode=dataset_evaluation_mode):
        # print(fold)
        for item_id, item in enumerate(dataset.train(fold)):
            if item['file'] not in files:
                files.append(item['file'])
        for item_id, item in enumerate(dataset.test(fold)):
            if item['file'] not in files:
                files.append(item['file'])

    return dataset


def process_parameters(params):

    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    # Paths
    params['path']['data'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['data'])
    params['path']['base'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['base'])

    return params

def restructure_asEvent_ann(dataset, fold, scene_label):

    ann = {}
    annDirPath = {}
    for item_id, item in enumerate(dataset.train(fold=fold, scene_label=scene_label)): # training for k-fold dataset
    # for item_id, item in enumerate(dataset.train(fold=0, scene_label=scene_label)): # training with all dataset for release

        # item:= 'file', 'scene_label', 'event_onset', 'event_offset', 'event_label'

        dirPath = os.path.split(item['file'])[0]
        filename = os.path.split(item['file'])[1]

        if filename not in ann:
            ann[filename] = {}
        if item['event_label'] not in ann[filename]:
            ann[filename][item['event_label']] = []
            
        ann[filename][item['event_label']].append((item['event_onset'], item['event_offset']))
        annDirPath[filename] = item['file'];

    return ann, annDirPath;

def make_data_posAndNeg(hop_length_seconds, feature_data, ann, audio_filename, data_positive, data_negative):
    for event_label in ann[audio_filename]:
        # print(ann[audio_filename][event_label])
        positive_mask = numpy.zeros((feature_data.shape[0]), dtype=bool)

        for event in ann[audio_filename][event_label]:
            # hop length: overrap window parameter
            start_frame = int(math.floor(event[0] / hop_length_seconds))
            stop_frame = int(math.ceil(event[1] / hop_length_seconds))

            if stop_frame > feature_data.shape[0]:
                stop_frame = feature_data.shape[0]

            positive_mask[start_frame:stop_frame] = True

        # Store positive examples
        if event_label not in data_positive:
            data_positive[event_label] = feature_data[positive_mask, :]
        else:
            data_positive[event_label] = numpy.vstack((data_positive[event_label], feature_data[positive_mask, :]))
            # data_positive[event_label] = numpy.hstack((data_positive[event_label], feature_data[positive_mask, :]))

        # Store negative examples
        if event_label not in data_negative:
            data_negative[event_label] = feature_data[~positive_mask, :]
        else:
            data_negative[event_label] = numpy.vstack((data_negative[event_label], feature_data[~positive_mask, :]))
            # data_negative[event_label] = numpy.hstack((data_negative[event_label], feature_data[~positive_mask, :]))

    return data_positive, data_negative;

def make_test_posAndNeg(audio_filename, event_label, test_positive, test_negative):

    # Store negative examples
    if '_neg_' in audio_filename:
        if event_label not in test_negative:
            test_negative[event_label] = feature_data
        else:
            test_negative[event_label] = numpy.vstack((test_negative[event_label], feature_data))
            # data_negative[event_label] = numpy.hstack((data_negative[event_label], feature_data[~positive_mask, :]))
    else:
        # Store positive examples
        if event_label not in test_positive:
            test_positive[event_label] = feature_data
        else:
            test_positive[event_label] = numpy.vstack((test_positive[event_label], feature_data))
            # data_positive[event_label] = numpy.hstack((data_positive[event_label], feature_data[positive_mask, :]))

    return test_positive, test_negative


def makeFeatureVec(params, y, fs):

    feature_data = feat_dr.extract_features(y=y,
                                  fs=fs,
                                  include_mfcc0=params['features']['include_mfcc0'],
                                  include_delta=params['features']['include_delta'],
                                  include_acceleration=params['features']['include_acceleration'],
                                  mfcc_params=params['features']['mfcc'],
                                  delta_params=params['features']['mfcc_delta'],
                                  acceleration_params=params['features']['mfcc_acceleration'],
                                  isMfcc=params['features']['isMfcc'])
                                  # isMfcc=True)
    
    mfcc_raw = feature_data['feat'];    
    feature_data = mfcc_raw


    return feature_data

########################################################################################
########################################################################################

# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, k, k, 1],
                          padding='SAME')


# Create model
def conv_net(x, size_windows, mel_bin, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, size_windows, mel_bin, 1]) # 2nd: # of windows, 3rd: # of mel-bin

    # print('conv')

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    conv2 = conv2d(conv1, weights['wc2'], biases['bc1'])
    conv2 = maxpool2d(conv2, k=2)
    conv2 = tf.nn.dropout(conv2, 0.5)

    conv3 = conv2d(conv2, weights['wc3'], biases['bc2'])
    conv4 = conv2d(conv3, weights['wc4'], biases['bc2'])
    conv4 = maxpool2d(conv4, k=2)
    conv4 = tf.nn.dropout(conv4, 0.5)
 
    # # Convolution Layer
    # conv5 = conv2d(conv4, weights['wc5'], biases['bc3'])
    # conv6 = conv2d(conv5, weights['wc6'], biases['bc3'])
    # conv6 = maxpool2d(conv6, k=2)
    # conv6 = tf.nn.dropout(conv6, 0.5)

    # conv7 = conv2d(conv6, weights['wc7'], biases['bc4'])
    # conv8 = conv2d(conv7, weights['wc8'], biases['bc4'])
    # conv8 = maxpool2d(conv8, k=2)
    # conv8 = tf.nn.dropout(conv8, 0.25)

    print(conv4)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv4, [-1, weights['wd1'].get_shape().as_list()[0]])
    # print(weights['wd1'].get_shape().as_list()[0])
    shape_test = weights['wd1'].get_shape().as_list()
    # print(shape_test)
    # print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, 0.5)

    # Output, class prediction
    pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return pred

def conv_net_naive(x, size_windows, mel_bin, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, size_windows, mel_bin, 1]) # 2nd: # of windows, 3rd: # of mel-bin

    # print('conv')

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=2)

    # print(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc1 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    # print(weights['wd1'].get_shape().as_list()[0])
    shape_test = weights['wd1'].get_shape().as_list()
    # print(shape_test)
    # print(fc1)
    fc1 = tf.add(tf.matmul(fc1, weights['wd1']), biases['bd1'])
    fc1 = tf.nn.relu(fc1)
    # Apply Dropout
    fc1 = tf.nn.dropout(fc1, dropout)

    # Output, class prediction
    pred = tf.add(tf.matmul(fc1, weights['out']), biases['out'])

    return pred

class Dataset(object):
    def __init__(self, data_positive, data_negative, test_positive, test_negative, num_class, n_input, test_eventLabel, one_hot=False):
    
        # data shape: (# of examples, width * height)
        self._data_positive = data_positive
        self._data_negative = data_negative

        # print(data_positive['drone'].shape)
        # print(data_negative['drone'].shape)
        # print(test_positive.shape)
        # print(test_negative['drone'].shape)

        # make test dataset
        if(len(test_positive) > 0):
            test_positive = test_positive[test_eventLabel]

            new_pos_len = (test_positive.shape[0] * test_positive.shape[1]) / n_input
            test_pos_row = test_positive.shape[0]
            discard_pos_len = test_pos_row % new_pos_len
            test_positive = test_positive[0 : (test_pos_row-discard_pos_len) ]
            print(test_positive.shape)
            print(discard_pos_len)
            test_pos_reshape = numpy.reshape(test_positive, (-1, n_input))
            test_label_pos = numpy.zeros((test_pos_reshape.shape[0], num_class))
            test_label_pos[:, 1] = 1

        if(len(test_negative) > 0):
            test_negative = test_negative[test_eventLabel]

            new_neg_len = (test_negative.shape[0] * test_negative.shape[1]) / n_input
            test_neg_row = test_negative.shape[0]
            discard_neg_len = test_neg_row % new_neg_len
            test_negative = test_negative[0 : (test_neg_row-discard_neg_len) ]
            test_neg_reshape = numpy.reshape(test_negative, (-1, n_input))
            test_label_neg = numpy.zeros((test_neg_reshape.shape[0], num_class))
            test_label_neg[:, 0] = 1

        if len(test_positive)>0 and len(test_negative) < 1 :
            test_data = test_pos_reshape
            test_label = test_label_pos
        elif len(test_positive) < 1 and len(test_negative) > 0 :
            test_data = test_neg_reshape
            test_label = test_label_neg
        else:
            test_data = numpy.vstack((test_pos_reshape, test_neg_reshape))
            test_label = numpy.vstack((test_label_pos, test_label_neg))

        self._test_data = test_data
        self._test_label = test_label


        self._epochs_completed = 0
        self._ind_p_epoch = 0
        self._ind_n_epoch = 0

    @property
    def data_positive(self):
        return self._data_positive

    @property
    def data_negative(self):
        return self._data_negative

    @property
    def test_data(self):
        return self._test_data

    @property
    def test_label(self):
        return self._test_label        

    @property
    def num_positive(self):
        return self._num_positive

    @property
    def num_negative(self):
        return self._num_negative

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self, num_window, batch_size, num_class, event_label):

        half_batch = num_window * batch_size / 2

        # pos for batch
        start_p = self._ind_p_epoch
        self._ind_p_epoch += half_batch
        if self._ind_p_epoch > len(self._data_positive[event_label]):
            self._epochs_completed += 1

            perm = numpy.arange(len(self._data_positive[event_label]))
            numpy.random.shuffle(perm)
            self._data_positive[event_label] = self._data_positive[event_label][perm]

            start_p = 0
            self._ind_p_epoch = half_batch

        end_p = self._ind_p_epoch

        # neg for batch
        start_n = self._ind_n_epoch
        self._ind_n_epoch += half_batch
        if self._ind_n_epoch > len(self._data_negative[event_label]):
            self._epochs_completed += 1

            perm = numpy.arange(len(self._data_negative[event_label]))
            numpy.random.shuffle(perm)
            self._data_negative[event_label] = self._data_negative[event_label][perm]

            start_n = 0
            self._ind_n_epoch = half_batch

        end_n = self._ind_n_epoch

        ##
        next_pos = self._data_positive[event_label][start_p:end_p]
        next_neg = self._data_negative[event_label][start_n:end_n]
        
        next_data = numpy.vstack((next_pos, next_neg))

        onehot_neg = numpy.zeros((batch_size/2, num_class))
        onehot_neg[:,0] = 1
        onehot_pos = numpy.zeros((batch_size/2, num_class))
        onehot_pos[:,1] = 1
        next_label = numpy.vstack((onehot_pos, onehot_neg))

        # print(next_data.shape)
        # print('next label size')
        # print(next_label.shape)

        return next_data, next_label



########################################################################################
########################################################################################


## prepare data handling
params = loadParam_manually.loadParam_manual()
params = process_parameters(params)

# prepare dataset from files
files = [];
dataset_evaluation_mode = 'folds';
num_folds = 3;
dataset = readDataset(params, files, dataset_evaluation_mode, num_folds);

# # Import MNIST data
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("data/", one_hot=True)

model_path=params['path']['models']
for scene_id, scene_label in enumerate(dataset.scene_labels):
    # scene_label: home, residual, playground
    # current_model_file = get_model_filename(fold=fold, scene_label=scene_label, path=model_path)

    # limit_train_file = 4;
    # cur_train_file = 0;

    # Restructure training data in to structure[files][events]
    ann, annDirPath = restructure_asEvent_ann(dataset, num_folds, scene_label);

    # Collect training examples
    hop_length_seconds=params['features']['hop_length_seconds']
    classifier_params=params['classifier']['parameters']

    # pre-iterating for normalizer part
    # normalizer = feat_dr.FeatureNormalizer();
    # feat_dr.make_Normalizer(params, dataset, normalizer, ann, annDirPath);

    data_positive = {}
    data_negative = {}
    # # loop for making  for each audio file
    for item_id, audio_filename in enumerate(ann):
    # for item_id, item in enumerate(dataset.train(fold=fold, scene_label=scene_label)):
    #     fileNum = fileNum + 1
    #     # print(fileNum)
    #     # print('-------------')
    #     sys.stdout.flush()

    #     # load audio and make feature vector
        audio_filepath = annDirPath[audio_filename]
        # print(dataset.relative_to_absolute_path(audio_filepath))
        y, fs = files_dr.load_audio(filename=dataset.relative_to_absolute_path(audio_filepath), mono=True, fs=params['features']['fs'])
        feature_data = makeFeatureVec(params, y, fs)

        # print(feature_data.shape)

        data_positive, data_negative = make_data_posAndNeg(hop_length_seconds, feature_data, ann, audio_filename, data_positive, data_negative)

    print(data_positive['drone'].shape)

    # prepare testing dataset
    path_testData = '/home/sdeva/workspace/d_d/data/testing/161116_testLowFreq'
    # path_testData = '/home/sdeva/workspace/d_d/data/testing/161006_southDaejeon'
    # path_testData = '/home/sdeva/workspace/d_d/data/testing/161013_playground'
    path_testData = '/home/sdeva/workspace/d_d/data/testing/temp5'
    list_test_file = []
    for curFile in os.listdir(path_testData):
        if (curFile.endswith(".wav")):
            list_test_file.append(os.path.join(path_testData, curFile))

    # read test data
    test_positive = {}
    test_negative = {}
    test_eventLabel = 'drone'
    for i in range(len(list_test_file)):
        audio_filepath = list_test_file[i]
        curAudio_filename = os.path.basename(audio_filepath)
        fs_input = params['features']['fs']
        if 'neg_' in curAudio_filename:
            fs_input = 44100
        y, fs = files_dr.load_audio(filename=audio_filepath, mono=True, fs=fs_input)
        feature_data = makeFeatureVec(params, y, fs)

        test_positive, test_negative = make_test_posAndNeg(curAudio_filename, test_eventLabel, test_positive, test_negative)


    #####################################################################################################################
    ########### After data preparation, training cnn

    step = 1
    # input_len = 6

    # for event in data_positive:
    #     data_positive[event] = data_positive[event].T
    #     data_negative[event] = data_negative[event].T

    # # test code for batch_data
    # print(data_positive['drone'].shape)
    # print(data_negative['drone'].shape)

    # for i in range(100):
    #     next_data, next_label = dataset.next_batch(batch_size, 'drone')

    #     print(next_data)
    #     print(next_label)

    # Parameters
    learning_rate = 0.001
    training_iters = 200000000
    batch_size = 128
    display_step = 2000

    # # Network Parameters
    mel_bin = params['features']['mfcc']['n_mels']
    num_mfcc = params['features']['mfcc']['n_mfcc'] * 2 - 1
    input_num_windows = int(params['features']['win_length_seconds'] / params['features']['hop_length_seconds']) # set as 2 when 50% overlapping
    network_num_window = 6 # actual_length = win_len * network_num_window
    n_input = mel_bin * input_num_windows * network_num_window # data input (img shape: (mel-bin, input_num_windows * network_num_window))
    # n_input = num_mfcc * input_num_windows * network_num_window # data input (img shape: (mel-bin, input_num_windows * network_num_window))
    n_classes = 2 # total classes (true or false
    dropout = 0.75 # Dropout, probability to keep units

    # make input dataset class
    dataset = Dataset(data_positive, data_negative, test_positive, test_negative, n_classes, n_input, test_eventLabel)

    print('input size: ' + str(n_input))

    # # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Store layers weight & bias
    # weights = {
    #     # 3x3 conv, 1 input, 32 outputs
    #     'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
    #     # 3x3 conv, 32 inputs, 64 outputs
    #     'wc2': tf.Variable(tf.random_normal([3, 3, 32, 64])),
    #     # fully connected, 7*7*64 inputs, 1024 outputs
    #     'wd1': tf.Variable(tf.random_normal([6*15*64, 1024])), # should be calculated by hand!!!
    #     # 1024 inputs, 10 outputs (class prediction)
    #     'out': tf.Variable(tf.random_normal([1024, n_classes]))
    # }

    weights = {
        # 3x3 conv, 1 input, 32 outputs
        'wc1': tf.Variable(tf.random_normal([3, 3, 1, 32])),
        'wc2': tf.Variable(tf.random_normal([3, 3, 32, 32])),
        # 3x3 conv, 32 inputs, 64 outputs
        'wc3': tf.Variable(tf.random_normal([3, 3, 32, 256])),
        'wc4': tf.Variable(tf.random_normal([3, 3, 256, 256])),

        'wc5': tf.Variable(tf.random_normal([3, 3, 64, 128])),
        'wc6': tf.Variable(tf.random_normal([3, 3, 128, 128])),

        'wc7': tf.Variable(tf.random_normal([3, 3, 128, 256])),
        'wc8': tf.Variable(tf.random_normal([3, 3, 256, 256])),


        # fully connected, 7*7*64 inputs, 1024 outputs
        'wd1': tf.Variable(tf.random_normal([3*10*256, 1024])), # should be calculated by hand!!!
        # 1024 inputs, 10 outputs (class prediction)
        'out': tf.Variable(tf.random_normal([1024, n_classes]))
    }

    biases = {
        'bc1': tf.Variable(tf.random_normal([32])),
        'bc2': tf.Variable(tf.random_normal([256])),
        'bc3': tf.Variable(tf.random_normal([128])),
        'bc4': tf.Variable(tf.random_normal([256])),
        'bd1': tf.Variable(tf.random_normal([1024])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # # Construct model
    # x = tf.reshape(x, shape=[-1, input_num_windows * network_num_window, mel_bin, 1]) # ???
    pred = conv_net(x, input_num_windows*network_num_window, mel_bin, weights, biases, keep_prob)
    # pred = conv_net(x, input_num_windows*network_num_window, num_mfcc, weights, biases, keep_prob)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    # init = tf.initialize_all_variables()
    init = tf.global_variables_initializer()
    loss = 10
    acc = 0.3

    saver = tf.train.Saver(max_to_keep=1000)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_points')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # pred label file
    filename_pred = 'label_pred_cnn.txt'
    file_pred = open(filename_pred, 'w')

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)
        step = 1

        saver.restore(sess, os.path.join('/home/sdeva/workspace/d_d/dd_python/check_points_cnn', 'cnn_2560000.ckpt'))

        # # Keep training until reach max iterations
        # while step * batch_size < training_iters:
        # # while loss > 1:
        #     batch_x, batch_y = dataset.next_batch(
        #         num_window=input_num_windows * network_num_window, 
        #         batch_size=batch_size, 
        #         num_class=n_classes, 
        #         event_label='drone')

        #     batch_x = numpy.reshape(batch_x, (-1, n_input))

        #     # Run optimization op (backprop)
        #     sess.run(optimizer, feed_dict={x: batch_x, y: batch_y,
        #                                    keep_prob: dropout})
        #     if step % display_step == 0:
        #         # Calculate batch loss and accuracy
        #         loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
        #                                                           y: batch_y,
        #                                                           keep_prob: 1.})
        #         print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
        #               "{:.6f}".format(loss) + ", Training Accuracy= " + \
        #               "{:.5f}".format(acc))

        #         next_test_data = dataset.test_data
        #         next_test_label = dataset.test_label

        #         test_loss, test_acc = sess.run([cost, accuracy], feed_dict={x: next_test_data,
        #                                   y: next_test_label})

        #         print("Minibatch Loss= " + \
        #               "{:.6f}".format(test_loss) + ", Cur Epoch Testing Accuracy= " + \
        #               "{:.5f}".format(test_acc))

        #         # save current model
        #         modelName_save = 'cnn_' + str(step*batch_size) + '.ckpt'
        #         modelPath = os.path.join(checkpoint_dir, modelName_save)
        #         saver.save(sess, modelPath)

        #     step += 1
        # print("Optimization Finished!")

        next_test_data = dataset.test_data
        next_test_label = dataset.test_label

        step = 0
        acc_avg = 0
        while step < 10:
            y_p = tf.argmax(pred, 1)
            start_time = time.time()
            acc, y_pred = sess.run([accuracy, y_p], feed_dict={x: next_test_data,
                                          y: next_test_label})
            elapsed_time = time.time() - start_time
            print(elapsed_time)

            print("Testing Accuracy: " + str(acc))
            acc_avg += acc

            y_true_tf = tf.argmax(next_test_label, 1)
            print(y_pred.shape)
            # print(next_test_label)
            y_true = y_true_tf.eval()
            # print(y_true)

            confu_mat = tf.contrib.metrics.confusion_matrix(y_pred, y_true, num_classes=2)
            c = confu_mat.eval()
            print(c)

            # file_pred.write(y_pred)
            numpy.savetxt(file_pred, y_pred, '%d')
            file_pred.write('\n')

            # print("f1_score", sk.metrics.f1_score(y_true, y_pred))

            # print("Final Testing Accuracy:", \
            #     sess.run(accuracy, feed_dict={x: next_test_data,
            #                                   y: next_test_label}))
            step += 1

        acc_avg = acc_avg / 10
        print(acc_avg)

        # # Calculate accuracy for test
        # print("Testing Accuracy:", \
        #     sess.run(accuracy, feed_dict={x: dataset._test_data,
        #                                   y: dataset._test_label,
        #                                   keep_prob: 1.}))

    file_pred.close()