# -*- coding: utf-8 -*-

from src_dd.ui import *
# from src.general import *

#from src.features import *
# from src.sound_event_detection import *
from src_dd.dataset import *
#from src.evaluation import *

import src_dd.features_drone as feat_dr
import src_dd.evaluation_drone as eval_dr
import src_dd.util_drone as util_dr
import src_dd.files as files_dr
import src_dd.modeling_manual_1 as modeling_manual
import src_dd.loadParam_manually as loadParam_manually

import numpy
import csv
import warnings
import argparse
import textwrap
import math
import re

import librosa
import scipy

import time
import datetime
from time import strftime
import sys

from sklearn import mixture
from sklearn import svm

import pickle

def run(argv):
    #### Training, Testing with GMM and Evaluation
    print('Start-----')
    sys.stdout.flush()

    #### read params
    # parameter_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), os.path.splitext(os.path.basename(__file__))[0]+'.yaml');
    # params = files_dr.load_parameters(parameter_file);
    params = loadParam_manually.loadParam_manual()
    params = process_parameters(params);
    # make_folders(params)

    # prepare dataset from files
    files = [];
    dataset_evaluation_mode = 'folds';
    num_folds = 3;
    dataset = readDataset(params, files, dataset_evaluation_mode, num_folds);

    # cover loop for changing parameters with experiment 
    exp_change_params = {}
    # exp_change_params['n_gmm'] = [15, 20, 25, 30, 40]
    exp_change_params['n_gmm'] = [13, 15, 20, 25, 30, 35, 40]
    # exp_change_params['n_gmm'] = [4,8,12,16,20]
    # exp_change_params['exp_n_mfcc'] = [13, 15, 20, 25, 30, 35]
    exp_change_params['exp_n_mfcc'] = [20, 30, 35]
    # exp_change_params['exp_n_mels'] = [10, 15, 20, 25, 30, 40, 45]
    exp_change_params['exp_n_mels'] = [40]
    # exp_change_params['exp_n_fft'] = [2048, 4096, 8192, 16384, 32768]
    # exp_change_params['exp_n_fft'] = [2048, 2048, 2048, 2048, 2048]
    # exp_change_params['exp_win_len'] = [40, 40, 40, 40, 40]

    # exp_change_params['minimum_event_length'] = [0.04, 0.08, 0.12, 0.16, 0.20]
    # exp_change_params['minimum_event_length'] = [0.16, 0.20]
    # exp_change_params['minimum_event_gap'] = [0.04, 0.08, 0.12, 0.16, 0.20]
    # exp_change_params['hop_length_seconds'] = [0.25, 0.5, 0.75, 1] # ratio for length

    # write result at each fold
    curTime = strftime("%Y-%m-%d_%H-%M")
    basepath_result = os.path.join(params['path']['base'], params['path']['results_'])
    if not os.path.isdir(basepath_result):
        os.makedirs(basepath_result)

    path_curResult = os.path.join(basepath_result, curTime)
    if not os.path.isdir(path_curResult):
        os.makedirs(path_curResult)

    for ind_exp in range(len(exp_change_params['n_gmm'])):
        # modify param for perforamnce test as different value (e.g, # of gmm, n_fft, # of mfcc)
        # params = modify_parameters(params, exp_change_params, ind_exp)
        params['classifier_parameters']['gmm']['n_components'] = exp_change_params['n_gmm'][ind_exp]
        # print("#_GMM: " + str(params['classifier_parameters']['gmm']['n_components']))

        # print("min event gap:" + str(params['detector']['minimum_event_gap']))

        # # main loop for fold
        # overall_metrics_per_scene = {}
        # processing_time = {}
        # result_summary = {}

        for ind_exp_j in range(len(exp_change_params['exp_n_mfcc'])):
            params['features']['mfcc']['n_mfcc'] = exp_change_params['exp_n_mfcc'][ind_exp_j]
            # print("#_MFCC: " + str(params['features']['mfcc']['n_mfcc']))

            for ind_exp_k in range(len(exp_change_params['exp_n_mels'])):
                if (exp_change_params['exp_n_mfcc'][ind_exp_j] > exp_change_params['exp_n_mels'][ind_exp_k]):
                    continue

                params['features']['mfcc']['n_mels'] = exp_change_params['exp_n_mels'][ind_exp_k]
                print(str(params['classifier_parameters']['gmm']['n_components']) +"\t" + str(params['features']['mfcc']['n_mfcc']) + "\t" + str(params['features']['mfcc']['n_mels']))

                # main loop for fold
                overall_metrics_per_scene = {}
                processing_time = {}
                result_summary = {}

                # fold: 1,2,3,4
                # Initialize model container (will be used for loading from saved file)
                model_container = {'models': {}};
                # normalizer container to save as scene_label
                normalizer_container = {};

                #### Training for model
                print('Training -------')
                sys.stdout.flush()

                # model_container = modeling_manual.setModelManually(params['classifier']['parameters'], model_container, 'drone')
                model_container, normalizer_container = training_drone(model_container, normalizer_container, dataset, params, 0, processing_time)
                # model_container = training_separate(model_container, params)
                
                #### Testing with trained model
                print('Testing ----------------')
                sys.stdout.flush()

                testing_drone(overall_metrics_per_scene, model_container, 
                    normalizer_container, dataset, params, 0, processing_time, path_curResult, result_summary, ind_exp)
                # end for fold

                # end for j
                print('')
                break
            
            print('')
            break

        print('')
        break
    # end for ind_exp

def process_parameters(params):
    """Parameter post-processing.

    Parameters
    ----------
    params : dict
        parameters in dict

    Returns
    -------
    params : dict
        processed parameters

    """

    params['features']['mfcc']['win_length'] = int(params['features']['win_length_seconds'] * params['features']['fs'])
    params['features']['mfcc']['hop_length'] = int(params['features']['hop_length_seconds'] * params['features']['fs'])

    # Copy parameters for current classifier method
    params['classifier']['parameters'] = params['classifier_parameters'][params['classifier']['method']]

    # Hash
    params['features']['hash'] = get_parameter_hash(params['features'])
    params['classifier']['hash'] = get_parameter_hash(params['classifier'])
    params['detector']['hash'] = get_parameter_hash(params['detector'])

    # Paths
    params['path']['data'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['data'])
    params['path']['base'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['base'])

    # Features
    params['path']['features_'] = params['path']['features']
    params['path']['features'] = os.path.join(params['path']['base'],
                                              params['path']['features'],
                                              params['features']['hash'])

    # Feature normalizers
    params['path']['feature_normalizers_'] = params['path']['feature_normalizers']
    params['path']['feature_normalizers'] = os.path.join(params['path']['base'],
                                                         params['path']['feature_normalizers'],
                                                         params['features']['hash'])

    # Models
    # Save parameters into folders to help manual browsing of files.
    params['path']['models_'] = params['path']['models']
    params['path']['models'] = os.path.join(params['path']['base'],
                                            params['path']['models'],
                                            params['features']['hash'],
                                            params['classifier']['hash'])

    # Results
    params['path']['results_'] = params['path']['results']
    params['path']['results'] = os.path.join(params['path']['base'],
                                             params['path']['results'],
                                             params['features']['hash'],
                                             params['classifier']['hash'],
                                             params['detector']['hash'])
    return params

def modify_parameters(params, exp_change_params, ind_exp):
    for exp_var in exp_change_params:
        if exp_var == 'n_gmm':
            params['classifier_parameters']['gmm']['n_components'] = exp_change_params['n_gmm'][ind_exp]
        if exp_var == 'exp_n_mfcc':
            params['features']['mfcc']['n_mfcc'] = exp_change_params['exp_n_mfcc'][ind_exp]
        if exp_var == 'exp_n_mels':
            params['features']['mfcc']['n_mels'] = exp_change_params['exp_n_mels'][ind_exp]
        if exp_var == 'exp_n_fft':
            params['features']['mfcc']['n_fft'] = exp_change_params['exp_n_fft'][ind_exp]
        if exp_var == 'exp_win_len':
            params['features']['mfcc']['win_length_seconds'] = exp_change_params['exp_win_len'][ind_exp]
            params['features']['mfcc']['hop_length_seconds'] = exp_change_params['exp_win_len'][ind_exp] / 2

    return params

def make_folders(params, parameter_filename='parameters.yaml'):
    """Create all needed folders, and saves parameters in yaml-file for easier manual browsing of data.

    Parameters
    ----------
    params : dict
        parameters in dict

    parameter_filename : str
        filename to save parameters used to generate the folder name

    Returns
    -------
    nothing

    """

    # Check that target path exists, create if not
    check_path(params['path']['features'])
    check_path(params['path']['feature_normalizers'])
    check_path(params['path']['models'])
    check_path(params['path']['results'])

    # Save parameters into folders to help manual browsing of files.

    # Features
    feature_parameter_filename = os.path.join(params['path']['features'], parameter_filename)
    if not os.path.isfile(feature_parameter_filename):
        files_dr.save_parameters(feature_parameter_filename, params['features'])

    # Feature normalizers
    feature_normalizer_parameter_filename = os.path.join(params['path']['feature_normalizers'], parameter_filename)
    if not os.path.isfile(feature_normalizer_parameter_filename):
        files_dr.save_parameters(feature_normalizer_parameter_filename, params['features'])

    # Models
    model_features_parameter_filename = os.path.join(params['path']['base'],
                                                     params['path']['models_'],
                                                     params['features']['hash'],
                                                     parameter_filename)
    if not os.path.isfile(model_features_parameter_filename):
        files_dr.save_parameters(model_features_parameter_filename, params['features'])

    model_models_parameter_filename = os.path.join(params['path']['base'],
                                                   params['path']['models_'],
                                                   params['features']['hash'],
                                                   params['classifier']['hash'],
                                                   parameter_filename)
    if not os.path.isfile(model_models_parameter_filename):
        files_dr.save_parameters(model_models_parameter_filename, params['classifier'])

    # Results
    # Save parameters into folders to help manual browsing of files.
    result_features_parameter_filename = os.path.join(params['path']['base'],
                                                      params['path']['results_'],
                                                      params['features']['hash'],
                                                      parameter_filename)
    if not os.path.isfile(result_features_parameter_filename):
        files_dr.save_parameters(result_features_parameter_filename, params['features'])

    result_models_parameter_filename = os.path.join(params['path']['base'],
                                                    params['path']['results_'],
                                                    params['features']['hash'],
                                                    params['classifier']['hash'],
                                                    parameter_filename)
    if not os.path.isfile(result_models_parameter_filename):
        files_dr.save_parameters(result_models_parameter_filename, params['classifier'])

    result_detector_parameter_filename = os.path.join(params['path']['base'],
                                                      params['path']['results_'],
                                                      params['features']['hash'],
                                                      params['classifier']['hash'],
                                                      params['detector']['hash'],
                                                      parameter_filename)
    if not os.path.isfile(result_detector_parameter_filename):
        files_dr.save_parameters(result_detector_parameter_filename, params['detector'])

def get_feature_filename(audio_file, path, extension='cpickle'):

    return os.path.join(path, 'sequence_' + os.path.splitext(audio_file)[0] + '.' + extension)


def get_feature_normalizer_filename(fold, scene_label, path, extension='cpickle'):

    return os.path.join(path, 'scale_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def get_model_filename(fold, scene_label, path, extension='cpickle'):

    return os.path.join(path, 'model_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def get_result_filename(fold, scene_label, path, extension='txt'):

    if fold == 0:
        return os.path.join(path, 'results_' + str(scene_label) + '.' + extension)
    else:
        return os.path.join(path, 'results_fold' + str(fold) + '_' + str(scene_label) + '.' + extension)


def readDataset(params, files, dataset_evaluation_mode, num_folds):
    # remind that scene label should be described manually in "@property def scene_labels" dataset.py
    # print params['path']['data']
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

    return dataset;

def restructure_asEvent_ann(dataset, fold, scene_label):

    ann = {}
    annDirPath = {}
    # for item_id, item in enumerate(dataset.train(fold=fold, scene_label=scene_label)): # training for k-fold dataset
    for item_id, item in enumerate(dataset.train(fold=0, scene_label=scene_label)): # training with all dataset for release

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

        # Store negative examples
        if event_label not in data_negative:
            data_negative[event_label] = feature_data[~positive_mask, :]
        else:
            data_negative[event_label] = numpy.vstack((data_negative[event_label], feature_data[~positive_mask, :]))

    return data_positive, data_negative;


def contiguous_regions(activity_array):

    # Find the changes in the activity_array
    change_indices = numpy.diff(activity_array).nonzero()[0]

    # Shift change_index with one, focus on frame after the change.
    change_indices += 1

    if activity_array[0]:
        # If the first element of activity_array is True add 0 at the beginning
        change_indices = numpy.r_[0, change_indices]

    if activity_array[-1]:
        # If the last element of activity_array is True, add the length of the array
        change_indices = numpy.r_[change_indices, activity_array.size]

    # Reshape the result into two columns
    return change_indices.reshape((-1, 2))


def postprocess_event_segments(event_segments, minimum_event_length=0.1, minimum_event_gap=0.1):

    # 1. remove short events
    event_results_1 = []
    for event in event_segments:
        if event[1]-event[0] >= minimum_event_length or isclose(event[1] - event[0], minimum_event_length):
            event_results_1.append((event[0], event[1]))

    if len(event_results_1):
        # 2. remove small gaps between events
        event_results_2 = []

        # Load first event into event buffer
        buffered_event_onset = event_results_1[0][0]
        buffered_event_offset = event_results_1[0][1]
        for i in range(1, len(event_results_1)):
            if event_results_1[i][0] - buffered_event_offset > minimum_event_gap or isclose(event_results_1[i][0] - buffered_event_offset, minimum_event_gap):
                # The gap between current event and the buffered is bigger than minimum event gap,
                # store event, and replace buffered event
                event_results_2.append((buffered_event_onset, buffered_event_offset))
                buffered_event_onset = event_results_1[i][0]
                buffered_event_offset = event_results_1[i][1]
            else:
                # The gap between current event and the buffered is smalle than minimum event gap,
                # extend the buffered event until the current offset
                buffered_event_offset = event_results_1[i][1]

        # Store last event from buffer
        event_results_2.append((buffered_event_onset, buffered_event_offset))

        return event_results_2
    else:
        return event_results_1

def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def training_separate(model_container, params):

    dataDir_pos = 'E:\\Data\\Sound\\concatSound_drone\\cur_dr_reduced'
    dataDir_neg = 'E:\\Data\\Sound\\concatSound_drone\\cur_bg'

    data_positive = makeData_separate(dataDir_pos, params)
    data_negative = makeData_separate(dataDir_neg, params)

    model_container = {'models': {}};
    data_positive = numpy.vstack(data_positive)
    data_negative = numpy.vstack(data_negative)

    print data_positive.shape
    print data_negative.shape

    event_label = 'drone'
    classifier_params=params['classifier']['parameters']
    model_container['models'][event_label] = {}
    model_container['models'][event_label]['positive'] = mixture.GMM(**classifier_params).fit(data_positive)
    model_container['models'][event_label]['negative'] = mixture.GMM(**classifier_params).fit(data_negative)


    return model_container

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

        mfcc_diff = numpy.zeros(shape=(mfcc_raw.shape[0], mfcc_raw.shape[1]-1))
        for i in range(mfcc_raw.shape[0]):
            curMfcc = mfcc_raw[i];
            curVec = [];
            for j in range(len(curMfcc)-1):
                curVec.append(curMfcc[j+1] - curMfcc[j])
            mfcc_diff[i] = curVec

        feature_data = numpy.hstack((mfcc_raw, mfcc_diff))

        
        data_train.append(feature_data)
        
    return data_train

# end training_separate

def makeFeatureVec(params, y, fs):

    feature_data = feat_dr.extract_features(y=y,
                                  fs=fs,
                                  include_mfcc0=params['features']['include_mfcc0'],
                                  include_delta=params['features']['include_delta'],
                                  include_acceleration=params['features']['include_acceleration'],
                                  mfcc_params=params['features']['mfcc'],
                                  delta_params=params['features']['mfcc_delta'],
                                  acceleration_params=params['features']['mfcc_acceleration'],
                                  isMfcc=True)
    
    mfcc_raw = feature_data['feat'];    
    feature_data = mfcc_raw

    return feature_data


def training_drone(model_container, normalizer_container, dataset, params, fold, processing_time):

    start_tr = time.clock()

    model_path=params['path']['models']
    for scene_id, scene_label in enumerate(dataset.scene_labels):
        # scene_label: home, residual, playground
        # current_model_file = get_model_filename(fold=fold, scene_label=scene_label, path=model_path)

        # limit_train_file = 4;
        # cur_train_file = 0;

        # Restructure training data in to structure[files][events]
        ann, annDirPath = restructure_asEvent_ann(dataset, fold, scene_label);

        # Collect training examples
        hop_length_seconds=params['features']['hop_length_seconds']
        # classifier_params=params['classifier']['parameters']
        classifier_params = params['classifier_parameters']['gmm']

        # pre-iterating for normalizer part
        # normalizer = feat_dr.FeatureNormalizer();
        # feat_dr.make_Normalizer(params, dataset, normalizer, ann, annDirPath);

        # len_audio_sum = 0
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
            fs_input = params['features']['fs']
            y, fs = files_dr.load_audio(filename=dataset.relative_to_absolute_path(audio_filepath), mono=True, fs=fs_input)
            feature_data = makeFeatureVec(params, y, fs)

 
            data_positive, data_negative = make_data_posAndNeg(hop_length_seconds, feature_data, ann, audio_filename, data_positive, data_negative);

        # Train models for each class
        # print("Model Learning Start")
        sys.stdout.flush()
        for event_label in data_positive:
            # e.g., 'drone'
            # print event_label
            # print(data_positive[event_label].shape)
            # print(data_negative[event_label].shape)

            model_container['models'][event_label] = {}
            model_container['models'][event_label]['positive'] = mixture.GaussianMixture(n_components=classifier_params['n_components'],
                                      covariance_type=classifier_params['covariance_type'],
                                      tol=classifier_params['tol'],
                                      reg_covar=0.0,
                                      max_iter=classifier_params['n_iter'],
                                      n_init=classifier_params['n_init'],
                                      init_params=classifier_params['init_params']
                                      ).fit(data_positive[event_label])
            model_container['models'][event_label]['negative'] = mixture.GaussianMixture(n_components=classifier_params['n_components'],
                                      covariance_type=classifier_params['covariance_type'],
                                      tol=classifier_params['tol'],
                                      reg_covar=0.0,
                                      max_iter=classifier_params['n_iter'],
                                      n_init=classifier_params['n_init'],
                                      init_params=classifier_params['init_params']
                                      ).fit(data_negative[event_label])
            
        # print("Model Learning End")
        sys.stdout.flush()            

        # store normalizer as scene_label
        # normalizer_container[scene_label] = normalizer

        # Save models
        # files_dr.save_data(current_model_file, model_container)

        # Save model as txt format for API version
        # writeModelInfo_Txt(model_container, event_label)


        break; # break for just one scene test

    end_tr = time.clock();
    pTime_tr = end_tr - start_tr
    processing_time['training'] = pTime_tr

    return model_container, normalizer_container

def writeModelInfo_Txt(model_container, event_label):
    # # Print trained model information (mean, var)
    # print(fold)
    # positive model print
    modelInfo = open('modelInfo.txt', 'w')
    # print('Pos model \n')
    # print(model_container['models'][event_label]['positive'].means_.shape)
    # print(model_container['models'][event_label]['positive'].means_.dtype)
    # print(type(model_container['models'][event_label]['positive'].converged_))
    # print(model_container['models'][event_label]['positive'].converged_)        
    # # print('\t'.join(map(str, model_container['models'][event_label]['positive'].means_)))
    # print('\n')
    # print('\t'.join(map(str, model_container['models'][event_label]['positive'].weights_)))
    # print('\n\n')
    # # negative model print
    # print('Neg model \n')
    # print(model_container['models'][event_label]['negative'].means_.shape)
    # print('\t'.join(map(str, model_container['models'][event_label]['negative'].means_)))
    # print('\n')
    # print('\t'.join(map(str, model_container['models'][event_label]['negative'].weights_)))
    # print('\n\n')

    modelInfo.write('Pos model\n')
    modelInfo.write('means\n')
    modelInfo.write(str(model_container['models'][event_label]['positive'].means_.shape))
    modelInfo.write('\n')
    str_mean = '\t'.join(map(str, model_container['models'][event_label]['positive'].means_))
    str_mean = makeTxt_ForVariable(str_mean)
    modelInfo.write(str_mean)
    modelInfo.write('\n')
    modelInfo.write('weights\n')
    modelInfo.write(str(model_container['models'][event_label]['positive'].weights_.shape))
    str_weights = '\t'.join(map(str, model_container['models'][event_label]['positive'].weights_))
    str_weights = makeTxt_ForVariable(str_weights)
    modelInfo.write(str_weights)
    modelInfo.write('\n')
    modelInfo.write('covar\n')
    modelInfo.write(str(model_container['models'][event_label]['positive'].covars_.shape))
    str_covar = '\t'.join(map(str, model_container['models'][event_label]['positive'].covars_))
    str_covar = makeTxt_ForVariable(str_covar)
    modelInfo.write(str_covar)
    modelInfo.write('\n\n')
    # negative model print
    modelInfo.write('Neg model\n')
    modelInfo.write('means\n')
    modelInfo.write(str(model_container['models'][event_label]['negative'].means_.shape))
    modelInfo.write('\n')
    str_mean = '\t'.join(map(str, model_container['models'][event_label]['negative'].means_))
    str_mean = makeTxt_ForVariable(str_mean)
    modelInfo.write(str_mean)
    modelInfo.write('\n')
    modelInfo.write('weights\n')
    modelInfo.write(str(model_container['models'][event_label]['negative'].weights_.shape))
    str_weights = '\t'.join(map(str, model_container['models'][event_label]['negative'].weights_))
    str_weights = makeTxt_ForVariable(str_weights)
    modelInfo.write(str_weights)
    modelInfo.write('\n')
    modelInfo.write('covar\n')
    modelInfo.write(str(model_container['models'][event_label]['negative'].covars_.shape))
    str_covar = '\t'.join(map(str, model_container['models'][event_label]['negative'].covars_))
    str_covar = makeTxt_ForVariable(str_covar)
    modelInfo.write(str_covar)
    modelInfo.write('\n\n')
    modelInfo.close()

def makeTxt_ForVariable(input_str):
    
    result_str = input_str.replace('[  ', '[ ')
    result_str = re.sub('\s{2,}', ', ', result_str)
    result_str = result_str.replace('\t', ', ')
    
    return result_str



def testing_drone(overall_metrics_per_scene, model_container, normalizer_container, dataset, params,
 fold, processing_time, path_curResult, result_summary, ind_exp):

     # param for testing
    detector_params = params['detector']
    feature_params = params['features']
    decision_threshold = 1.0;

    measure_list = {}
    measure_list['f-score'] = {}
    measure_list['precision'] = {}
    measure_list['recall'] = {}

    # dataDir = '/home/sdeva/workspace/d_d/data/testing/161116_testLowFreq'
    # dataDir = '/home/sdeva/workspace/d_d/data/testing/161006_southDaejeon'
    # dataDir = '/home/sdeva/workspace/d_d/data/testing/161013_playground'
    dataDir = '/home/sdeva/workspace/d_d/data/testing/temp4'
    # dataDir = '/home/sdeva/workspace/d_d/data/cur_bg'
    list_test_file = []
    for curFile in os.listdir(dataDir):
        if (curFile.endswith(".wav")):
            list_test_file.append(os.path.join(dataDir, curFile))

    # measure store
    list_accuracy = []
    time_total_sum = 0

    dtime_sum = 0
    confu_mat = numpy.zeros(shape=(2,2))
    len_audio_sum = 0
    for i in range(len(list_test_file)):
        audio_filepath = list_test_file[i]
        # print(audio_filepath)

        # load audio and make feature vector
        fs_input = params['features']['fs']
        if 'neg_' in audio_filepath:
            fs_input = 44100
            # print('!!')

        # print('@@')
        # start_time = time.time()
        y, fs = files_dr.load_audio(filename=audio_filepath, mono=True, fs=fs_input)

        # elapsed_time = time.time() - start_time
        # print(elapsed_time)

        # y, fs = files_dr.load_audio(filename=dataset.relative_to_absolute_path(audio_file), mono=True, fs=params['features']['fs'])
        
        # with open('data_dll_positive.txt') as f_data:
        #     y = f_data.readlines()

        # y = [s.replace("\n", "") for s in y ]
        # y = [float(val) for val in y]
        
        # print('##')
        # start_time = time.time()
        feature_data = makeFeatureVec(params, y, fs)

        # elapsed_time = time.time() - start_time
        # print(elapsed_time)

        len_audio = len(feature_data) * feature_params['hop_length_seconds']
        print(len_audio)
        len_audio_sum += len_audio

        sys.stdout.flush()

        #
        # print("Feature time: " + str(end_feat - start_feat))
        # print(feature_data.shape)

        # params for prediction
        hop_length_seconds = feature_params['hop_length_seconds'];
        minimum_event_length = detector_params['minimum_event_length'];
        minimum_event_gap = detector_params['minimum_event_gap'];

        # results = [];
        # current_file_results = [];

        # measure performance w.r.t files
        # metrics_eval_segment_drone = eval_dr.Metrics_Eval_Segment_Drone(class_list=dataset.event_labels(scene_label=scene_label))
        for event_label in model_container['models']:
            # print(event_label)

            # # Get bic
            # bic_pos = model_container['models'][event_label]['positive'].bic(feature_data)
            # bic_neg = model_container['models'][event_label]['negative'].bic(feature_data)
            # list_bic_pos.append(bic_pos)
            # list_bic_neg.append(bic_neg)

            start_time = time.time()

            # positive = model_container['models'][event_label]['positive'].score_samples(feature_data)[0];
            # negative = model_container['models'][event_label]['negative'].score_samples(feature_data)[0];
            positive = model_container['models'][event_label]['positive'].score_samples(feature_data);
            negative = model_container['models'][event_label]['negative'].score_samples(feature_data);
            likelihood_ratio = positive - negative
            event_activity = likelihood_ratio > decision_threshold


            elapsed_time = time.time() - start_time

            print(elapsed_time)

            # print likelihood_ratio
            # print(event_activity)
            # print(likelihood_ratio)
            # print(numpy.amax(likelihood_ratio, axis=0))

            # event_activity = model_container['models'][event_label]['svm'].predict(feature_data)

            # Find contiguous segments and convert frame-ids into times
            event_segments = contiguous_regions(event_activity) * hop_length_seconds

            # Preprocess the event segments
            event_segments = postprocess_event_segments(event_segments=event_segments,
                                                       minimum_event_length=minimum_event_length,
                                                       minimum_event_gap=minimum_event_gap)

            ## Debuging print for event segmentation result
            # print 'pos'
            # print positive
            # print 'neg'
            # print negative
            # print event_segments
            # for event in event_segments:
            #     # results.append((event[0], event[1], event_label))
            #     # likelihoods_pos = [x for x in likelihood_ratio if x > 0]
            #     current_file_results.append(
            #         {'file': audio_file,
            #          'event_onset': event[0],
            #          'event_offset': event[1],
            #          'event_label': event_label,
            #          }
            #     )

            totalTime_event = 0
            for event in event_segments:
                # print(event)
                totalTime_event = totalTime_event + (event[1] - event[0])

            totalTime = float(len(event_activity)) * hop_length_seconds
            accuracy = 0
            detected_time = 0

            if "neg_" in os.path.basename(audio_filepath):
                detected_time = totalTime - totalTime_event
                accuracy = detected_time / totalTime
                confu_mat[1, 0] = confu_mat[1, 0] + totalTime_event

            # elif "_pos" in os.path.basename(audio_filepath):
            else:
                detected_time = totalTime_event
                accuracy =  detected_time / totalTime
                confu_mat[0, 1] = confu_mat[0, 1] + (totalTime - totalTime_event)
                confu_mat[1, 1] = confu_mat[1, 1] + totalTime_event


            firstDetected = -1
            if len(event_segments) > 0:
                firstDetected = event_segments[0][0]

            # evaluate
            audio_filename = os.path.basename(audio_filepath)
            print(audio_filename + "\t" + str(firstDetected) + "\t\t" + str(len(event_segments)) + "\t\t" + str(totalTime_event) + "\t\t" + str(totalTime) + "\t\t" + str(accuracy))

            list_accuracy.append(accuracy)

            # handling as total time instead of files
            time_total_sum = time_total_sum + totalTime
            dtime_sum = dtime_sum + detected_time

    acc_files = sum(list_accuracy) / float(len(list_accuracy))
    print("Total Accuracy:" + "\t" + str(acc_files))

    acc_time = dtime_sum / time_total_sum
    print("Acc in Total Time:" + str(acc_time))

    confu_mat = confu_mat / time_total_sum
    print(confu_mat)
    prec = confu_mat[1,1] / (confu_mat[1,1] + confu_mat[1,0])
    reca = confu_mat[1,1] / (confu_mat[1,1] + confu_mat[0,1])
    fsco = 2*prec*reca / (prec+reca)

    print('precision' + '\t' + str(prec))
    print('recall' + '\t' + str(reca))
    print('fscore' + '\t' + str(fsco))

    print('sum' + '\t' + str(len_audio_sum))

    return

def writeResult(path_curResult, dataset, result_summary, processing_time, params, ind_exp):
    # write summary of experiment
    extension = 'txt'
    path_rSummary = os.path.join(path_curResult, 'summary_result' + '_' + 'e' + str(ind_exp+1) + '.' + extension)
    # print path_rSummary
    file_rSummary = open(path_rSummary, 'w')

    # write performance
    file_rSummary.write('**** Performance' + '\n')
    for scene_id, scene_label in enumerate(dataset.scene_labels):
        fscore_sum = 0
        for result_fold in result_summary[scene_label]:
            curFscore_fold = result_summary[scene_label][result_fold]
            file_rSummary.write('fold' + str(result_fold) + ':' + '\t' + scene_label + '\t' + str(curFscore_fold) + '\n')
            fscore_sum = fscore_sum + curFscore_fold
        print ('Avg fscore:' + '\t' + str(result_summary[scene_label]))
        overall_fscore_scene = fscore_sum / len(result_summary[scene_label])
        file_rSummary.write('Avg f-score' + '\t' + scene_label + '\t' + str(overall_fscore_scene) + '\n')
        file_rSummary.write('\n')

    # write processing time
    file_rSummary.write('**** Processing Time' + '\n')
    file_rSummary.write('feature:' + '\t' + str(processing_time['feature']) + '\n')
    file_rSummary.write('prediction:' + '\t' + str(processing_time['prediction']) + '\n')
    # file_rSummary.write('training:' + '\t' + str(processing_time['training']) + '\n')
    file_rSummary.write('testing:' + '\t' + str(processing_time['testing']) + '\n')
    file_rSummary.write('\n')

    # write parameters (n_fft, n_mfcc, n_mels, win_length, fmin, fmax, n_gmm, k-fold, )
    file_rSummary.write('**** Parameters' + '\n')
    file_rSummary.write('n_fft:' + '\t' + str(params['features']['mfcc']['n_fft']) + '\n')
    file_rSummary.write('n_mfcc:' + '\t' + str(params['features']['mfcc']['n_mfcc']) + '\n')
    file_rSummary.write('n_mels:' + '\t' + str(params['features']['mfcc']['n_mels']) + '\n')
    file_rSummary.write('win_length:' + '\t' + str(params['features']['win_length_seconds']) + '\n')
    file_rSummary.write('fmin:' + '\t' + str(params['features']['mfcc']['fmin']) + '\n')
    file_rSummary.write('fmax:' + '\t' + str(params['features']['mfcc']['fmax']) + '\n')
    file_rSummary.write('n_gmm:' + '\t' + str(params['classifier_parameters']['gmm']['n_components']) + '\n')
    # file_rSummary.write('k-fold:' + '\t' + params['features']['mfcc']['n_fft'] + '\n')

    file_rSummary.write('\n')


    file_rSummary.close();

#######################################################
#######################################################

if __name__ == "__main__":
    try:
        sys.exit(run(sys.argv))
    except (ValueError, IOError) as e:
        sys.exit(e)