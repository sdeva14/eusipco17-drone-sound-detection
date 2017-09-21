'''
A Convolutional Network implementation example using TensorFlow library.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
'''

from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell

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

import sklearn as sk


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
    if 'neg_' in audio_filename:
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
                                  # isMfcc=params['features']['isMfcc'])
                                  isMfcc=False)
    
    mfcc_raw = feature_data['feat'];       
    feature_data = mfcc_raw


    return feature_data

def training_separate(params, data_positive, data_negative):

    dataDir_pos = '/home/sdeva/workspace/d_d/data/testing/temp2'
    dataDir_neg = '/home/sdeva/workspace/d_d/data/cur_bg'

    event_label = 'drone'

    data_positive[event_label] = makeData_separate(dataDir_pos, params)
    data_negative[event_label] = makeData_separate(dataDir_neg, params)

    data_positive[event_label] = numpy.vstack(data_positive[event_label])
    data_negative[event_label] = numpy.vstack(data_negative[event_label])

    print(data_positive[event_label].shape)
    print(data_negative[event_label].shape)


    return data_positive, data_negative

def makeData_separate(dataDir, params):
    list_train_file = []
    for curFile in os.listdir(dataDir):
        if (curFile.endswith(".wav")):
            list_train_file.append(os.path.join(dataDir, curFile))

    data_train = []
    for i in range(len(list_train_file)):
        audio_filepath = list_train_file[i]

        y, fs = files_dr.load_audio(filename=audio_filepath, mono=True, fs=params['features']['fs'])
        feature_data = feat_dr.extract_features(y=y,
                                              fs=fs,
                                              include_mfcc0=params['features']['include_mfcc0'],
                                              include_delta=params['features']['include_delta'],
                                              include_acceleration=params['features']['include_acceleration'],
                                              mfcc_params=params['features']['mfcc'],
                                              delta_params=params['features']['mfcc_delta'],
                                              acceleration_params=params['features']['mfcc_acceleration'],
                                              isMfcc=params['features']['isMfcc'])
            
        mfcc_raw = feature_data['feat'];
        
        data_train.append(mfcc_raw)
        
    return data_train

########################################################################################
########################################################################################

class Dataset(object):
    def __init__(self, data_positive, data_negative, test_positive, test_negative, num_class, n_input, n_steps, one_hot=False):
        # data shape: (# of examples, width * height)
        self._data_positive = data_positive
        self._data_negative = data_negative

        print('##')
        # print(data_positive['drone'].shape)
        # print(data_negative['drone'].shape)

        # print(test_negative['drone'].shape)

        if(len(test_positive) > 0):
            test_positive = test_positive[test_eventLabel]
            new_pos_len = (test_positive.shape[0] * test_positive.shape[1]) / (n_input * n_steps)
            test_pos_row = test_positive.shape[0]
            discard_pos_len = test_pos_row % new_pos_len
            test_positive = test_positive[0 : (test_pos_row-discard_pos_len) ]
            test_pos_reshape = numpy.reshape(test_positive, (-1, n_input * n_steps))
            test_label_pos = numpy.zeros((test_pos_reshape.shape[0], num_class))
            test_label_pos[:, 1] = 1
            test_pos_reshape = numpy.reshape(test_positive, (-1, n_steps, n_input))

        if(len(test_negative) > 0):
            test_negative = test_negative[test_eventLabel]
            new_neg_len = (test_negative.shape[0] * test_negative.shape[1]) / (n_input * n_steps)
            test_neg_row = test_negative.shape[0]
            discard_neg_len = test_neg_row % new_neg_len
            test_negative = test_negative[0 : (test_neg_row-discard_neg_len) ]
            test_neg_reshape = numpy.reshape(test_negative, (-1, n_input * n_steps))
            test_label_neg = numpy.zeros((test_neg_reshape.shape[0], num_class))
            test_neg_reshape = numpy.reshape(test_negative, (-1, n_steps, n_input))
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

        print('@@')
        print(test_data.shape)

        # self._test_data = test_data.reshape((-1, n_steps, n_input))
        self._test_data = test_data
        self._test_label = test_label

        print('!!')
        print(data_positive['drone'].shape)
        print(data_negative['drone'].shape)
        print('@@')
        print(self._test_data.shape)


        # self._test_data = test_data
        # self._test_label = test_label

        self._epochs_completed = 0
        self._ind_p_epoch = 0
        self._ind_n_epoch = 0

    def shuffle_data(self):
        perm = numpy.arange(len(self._data_positive['drone']))
        numpy.random.shuffle(perm)
        self._data_positive['drone'] = self._data_positive['drone'][perm]

        perm = numpy.arange(len(self._data_negative['drone']))
        numpy.random.shuffle(perm)
        self._data_negative['drone'] = self._data_negative['drone'][perm]


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
        # next_label = numpy.vstack((onehot_neg, onehot_pos))
        next_label = numpy.vstack((onehot_pos, onehot_neg))

        # print(next_data.shape)
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
    
    # training from data folder
    for item_id, audio_filename in enumerate(ann):
        audio_filepath = annDirPath[audio_filename]
        y, fs = files_dr.load_audio(filename=dataset.relative_to_absolute_path(audio_filepath), mono=True, fs=params['features']['fs'])
        feature_data = makeFeatureVec(params, y, fs)

        data_positive, data_negative = make_data_posAndNeg(hop_length_seconds, feature_data, ann, audio_filename, data_positive, data_negative)

    # train data for separate
    # data_positive, data_negative = training_separate(params, data_positive, data_negative)


    # prepare testing dataset
    # path_testData = '/home/sdeva/workspace/d_d/data/testing/161116_testLowFreq'
    # path_testData = '/home/sdeva/workspace/d_d/data/testing/161006_southDaejeon'
    # path_testData = '/home/sdeva/workspace/d_d/data/testing/161013_playground'
    path_testData = '/home/sdeva/workspace/d_d/data/testing/temp5'
    # path_testData = '/home/sdeva/workspace/d_d/data/cur_bg'

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

    # add more data manually
    # print('!!')
    # del data_positive['drone']
    # # print(data_positive['drone'].shape)
    # path_moreData = '/home/sdeva/workspace/d_d/data/testing/temp2'
    # list_more_file = []
    # for curFile in os.listdir(path_moreData):
    #     if (curFile.endswith(".wav")):
    #         list_more_file.append(os.path.join(path_moreData, curFile))    
    # for i in range(len(list_more_file)):
    #     audio_filepath = list_more_file[i]
    #     curAudio_filename = os.path.basename(audio_filepath)
    #     y, fs = files_dr.load_audio(filename=audio_filepath, mono=True, fs=params['features']['fs'])
    #     feature_data = makeFeatureVec(params, y, fs)   
        
    #     data_positive, data_negative = make_test_posAndNeg(curAudio_filename, test_eventLabel, data_positive, data_negative)         
    # print('!!')
    # print(data_positive['drone'].shape)
    # data_negative['drone'] = data_negative['drone'][1:14000]

    #####################################################################################################################
    ########### After data preparation, training cnn


    # Parameters
    step = 1
    learning_rate = 0.00005
    # learning_rate = 0.0001
    training_iters = 20000000
    batch_size = 64
    display_step = 10000

    # lr=0.00005, num_window = 6

    # tf define
    # tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")


    # # Network Parameters
    mel_bin = params['features']['mfcc']['n_mels']
    num_mfcc = params['features']['mfcc']['n_mfcc'] - 1
    input_num_windows = int(params['features']['win_length_seconds'] / params['features']['hop_length_seconds'])
    network_num_window = 6 # actual_length = win_len * network_num_window

    # rnn parameters
    n_input = mel_bin
    # n_input = num_mfcc
    n_steps = network_num_window * input_num_windows
    n_hidden = 300 # hidden layer num of features
    num_multi_rnn = 3
    
    n_classes = 2 # total classes (true or false
    # dropout = 0.4 # Dropout, probability to keep units
    dropout = tf.placeholder(tf.float32)

    # make input dataset class
    test_eventLabel = 'drone'
    dataset = Dataset(data_positive, data_negative, test_positive, test_negative, n_classes, n_input, n_steps, test_eventLabel)
    dataset.shuffle_data()

    print('input size: ' + str(n_input))

    # # tf Graph input
    x = tf.placeholder(tf.float32, [None, n_steps, n_input])
    y = tf.placeholder(tf.float32, [None, n_classes])
    # keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

    # Define weights
    weights = {
        # 'out': tf.Variable(tf.random_normal([n_hidden, n_classes]))
        'out': tf.Variable(tf.random_normal([n_hidden * 2, n_classes]))
    }
    biases = {
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    def RNN(x, weights, biases):
        # Prepare data shape to match `rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshaping to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define a lstm cell with tensorflow
        lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def BiRNN(x, weights, biases):
        # Prepare data shape to match `bidirectional_rnn` function requirements
        # Current data input shape: (batch_size, n_steps, n_input)
        # Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # Define lstm cells with tensorflow
        # Forward direction cell
        lstm_fw_cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)
        # Backward direction cell
        lstm_bw_cell = rnn_cell.LSTMCell(n_hidden, forget_bias=1.0)

        # Get lstm cell output
        try:
            outputs, _, _ = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                                  dtype=tf.float32)
        except Exception: # Old TensorFlow version only returns outputs not states
            outputs = rnn.bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x,
                                            dtype=tf.float32)

        # Linear activation, using rnn inner loop last output
        return tf.matmul(outputs[-1], weights['out']) + biases['out']

    def multiRNN(x, weights, biases, num_layers):
        # Permuting batch_size and n_steps
        x = tf.transpose(x, [1, 0, 2])
        # Reshape to (n_steps*batch_size, n_input)
        x = tf.reshape(x, [-1, n_input])
        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        x = tf.split(0, n_steps, x)

        # # x = tf.transpose(tf.pack(x), perm=[1,0,2])
        # print(x)

        initializer = tf.random_uniform_initializer(-1, 1)

        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden)
        cell = tf.nn.rnn_cell.DropoutWrapper(cell=cell, output_keep_prob=0.5)
        cell = tf.nn.rnn_cell.MultiRNNCell(cells=[cell] * num_layers)

        cell2 = tf.nn.rnn_cell.BasicLSTMCell(num_units=n_hidden)
        cell2 = tf.nn.rnn_cell.DropoutWrapper(cell=cell2, output_keep_prob=0.5)
        cell2 = tf.nn.rnn_cell.MultiRNNCell(cells=[cell2] * num_layers)

        # init_state = multicell.zero_state(batch_size, tf.float32)

        # # Get lstm cell output
        # try:
        #     outputs, _, _ = rnn.bidirectional_rnn(cell, cell, x,
        #                                           dtype=tf.float32)
        # except Exception: # Old TensorFlow version only returns outputs not states
        #     outputs = rnn.bidirectional_rnn(cell, cell, x,
        #                                     dtype=tf.float32)
        
        # try:
        #     outputs, _, _ = rnn.rnn(multicell, x, dtype=tf.float32)
        # except Exception: # Old TensorFlow version only returns outputs not states
        #     outputs = rnn.rnn(multicell, x, dtype=tf.float32)
        output, _, _ = rnn.bidirectional_rnn(cell, cell2, x, dtype=tf.float32)

        # output = tf.transpose(output, [1, 0, 2])

        # Linear activation, using rnn inner loop last output
        # return tf.matmul(rnn_outputs[-1], weights['out']) + biases['out']
        # return tf.matmul(rnn_outputs[-1], W) + b

        # output = tf.transpose(output, [1, 0, 2])
        # last = tf.gather(output, int(output.get_shape()[0]) - 1)
        # # Softmax layer.
        # weight = tf.Variable(tf.truncated_normal([n_hidden, y.get_shape()[1]], stddev=0.01))
        # bias = tf.Variable(tf.constant(0.1, shape=[y.get_shape()[1]]))

        # prediction = tf.nn.softmax(tf.matmul(last, weight) + bias)
        pred = tf.matmul(output[-1], weights['out']) + biases['out']

        return pred
        
        # logits = [tf.matmul(rnn_output, W) + b for rnn_output in rnn_outputs]
        # return logits





    # # Construct model
    # x = tf.reshape(x, shape=[-1, input_num_windows * network_num_window, mel_bin, 1]) # ???
    # pred = RNN(x, weights, biases)
    # pred = BiRNN(x, weights, biases)
    pred = multiRNN(x, weights, biases, num_multi_rnn)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    # Evaluate model
    correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    # Initializing the variables
    init = tf.initialize_all_variables()

    saver = tf.train.Saver(max_to_keep=1000)
    checkpoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'check_points')
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # pred label file
    filename_pred = 'label_pred_rnn.txt'
    file_pred = open(filename_pred, 'w')

    step_epoch = 1
    # Launch the graph
    with tf.Session() as sess:

        config = tf.ConfigProto(
        device_count = {'GPU': 0}
        )
        sess = tf.Session(config=config)
    
        sess.run(init)

        saver.restore(sess, os.path.join('/home/sdeva/workspace/d_d/dd_python/check_points_rnn_2', 'birnn_5520000.ckpt'))

        # Training
        step = 1
        # # Keep training until reach max iterations
        # while step * batch_size < training_iters :
        #     batch_x, batch_y = dataset.next_batch(
        #         num_window=input_num_windows * network_num_window, 
        #         batch_size=batch_size, 
        #         num_class=n_classes, 
        #         event_label='drone')

        #     # batch_x = numpy.reshape(batch_x, (-1, 720))
        #     # print(batch_x.shape)
        #     # print(batch_x.shape[0] * batch_x.shape[1])
        #     # print(batch_size * n_input)
        #     batch_x = batch_x.reshape((batch_size, n_steps, n_input))

        #     # Run optimization op (backprop)
        #     sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        #     if step*batch_size % display_step == 0:
        #     # dataLen = len(data_positive['drone']) + len(data_negative['drone'])
        #     # if (step * batch_size) / dataLen > step_epoch:
        #         # Calculate batch loss and accuracy
        #         loss, acc = sess.run([cost, accuracy], feed_dict={x: batch_x,
        #                                                           y: batch_y})
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

        #         # label = sess.run(correct_pred, feed_dict={x: next_test_data,
        #         #                           y: next_test_label})

        #         # print(label)
        #         # print(len(label))

        #         step_epoch += 1

        #         # save current model
        #         modelName_save = 'birnn_' + str(step*batch_size) + '.ckpt'
        #         modelPath = os.path.join(checkpoint_dir, modelName_save)
        #         saver.save(sess, modelPath)

        #     step += 1

        # # Restore and testing
        # next_test_data = dataset.test_data
        # next_test_label = dataset.test_label
        # saver.restore(sess, os.path.join('/home/sdeva/workspace/d_d/dd_python/cp_rnn', 'birnn_5680000.ckpt'))

        # print("Optimization Finished!")

        # Calculate accuracy for test
        print(dataset._test_data.shape)
        # next_test_data = dataset._test_data.reshape((-1, n_steps, n_input))
        next_test_data = dataset.test_data
        next_test_label = dataset.test_label

        print(next_test_data.shape)
        print(next_test_label.shape)

        step = 0
        avg_acc = 0
        while step < 10:
            y_p = tf.argmax(pred, 1)

            start_time = time.time()
            acc, y_pred = sess.run([accuracy, y_p], feed_dict={x: next_test_data,
                                          y: next_test_label})
            elapsed_time = time.time() - start_time
            print(elapsed_time)

            print("Final Testing Accuracy: " + str(acc))
            avg_acc += acc

            y_true_tf = tf.argmax(next_test_label, 1)
            print(y_pred.shape)
            # print(next_test_label)
            y_true = y_true_tf.eval()
            # print(y_true)

            confu_mat = tf.contrib.metrics.confusion_matrix(y_pred, y_true, num_classes=2)
            c = confu_mat.eval()
            # c = c[0]
            print(c)

            # prec = c[1,1] / (c[1,1] + c[1,0])
            # reca = c[1,1] / (c[1,1] + c[0,1])
            # fsco = 2*prec*reca / (prec+reca)

            # print('precision' + '\t' + str(prec))
            # print('recall' + '\t' + str(reca))
            # print('fscore' + '\t' + str(fsco))

            # file_pred.write(y_pred)
            numpy.savetxt(file_pred, y_pred, '%d')
            file_pred.write('\n')

            # print("f1_score", sk.metrics.f1_score(y_true, y_pred))

            # print("Final Testing Accuracy:", \
            #     sess.run(accuracy, feed_dict={x: next_test_data,
            #                                   y: next_test_label}))
            step += 1

        avg_acc = avg_acc / 10
        print(avg_acc)


    file_pred.close()