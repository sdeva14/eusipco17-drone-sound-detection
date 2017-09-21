import src_dd.files as files_dr
import src_dd.features_drone as feat_dr
import src_dd.loadParam_manually as loadParam_manually


import librosa
import scipy
import numpy
from sklearn import mixture
import sys


import math
import re
import os

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

    # Paths
    params['path']['data'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['data'])
    params['path']['base'] = os.path.join(os.path.dirname(os.path.realpath(__file__)), params['path']['base'])

    return params

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
    
    mfcc_raw = feature_data['feat'];

    # mfcc gradient vector
    # mfcc_diff = numpy.zeros(shape=(mfcc_raw.shape[0], mfcc_raw.shape[1]-2))
    # for i in range(mfcc_raw.shape[0]):
    #     curMfcc = mfcc_raw[i];
    #     curVec = [];
    #     for j in range(1, len(curMfcc)-1):
    #         curVec.append(curMfcc[j+1] - curMfcc[j])
    #     mfcc_diff[i] = curVec
    # feature_data = mfcc_diff

    # 2nd order mfcc
    # feature_data = mfcc_raw[:, 1:]

    feature_data = mfcc_raw

    # print(len(y))
    # print(len(feature_data))        


    return feature_data

#################################################################################
#################################################################################

if __name__ == "__main__":
    # dataDir = 'E:\\Data\Sound\\161111_smi_wav'
    # dataDir_dr = 'E:\\Data\\Sound\\concatSound_drone\\cur_dr'
    # dataDir_bg = 'E:\\Data\\Sound\\concatSound_drone\\cur_bg_reduced'
    dataDir = 'E:\\Data\\Sound\\161116_testLowFreq'
    list_train_file = []
    for curFile in os.listdir(dataDir):
        if (curFile.endswith(".wav")):
            list_train_file.append(os.path.join(dataDir, curFile))
    # for curFile in os.listdir(dataDir_dr):
    #     if (curFile.endswith(".wav")):
    #         list_train_file.append(os.path.join(dataDir_dr, curFile))
    # for curFile in os.listdir(dataDir_bg):
    #     if (curFile.endswith(".wav")):
    #         list_train_file.append(os.path.join(dataDir_bg, curFile))

    params = loadParam_manually.loadParam_manual()
    params = process_parameters(params);
    classifier_params=params['classifier']['parameters']

    data_positive = []
    data_negative = []
    model_container = {'models': {}};
    for i in range(len(list_train_file)):
        audio_filepath = list_train_file[i]

        y, fs = files_dr.load_audio(filename=audio_filepath, mono=True, fs=params['features']['fs'])
        feature_data = makeFeatureVec(params, y, fs)

        if 'pos' in os.path.basename(audio_filepath) or 'Appro_' in os.path.basename(audio_filepath)  or 'Hov_' in os.path.basename(audio_filepath):
            data_positive.append(feature_data)

        if 'neg' in os.path.basename(audio_filepath):
            data_negative.append(feature_data)
        
        # print os.path.basename(audio_filepath)
        # print feature_data

    data_positive = numpy.vstack(data_positive)
    data_negative = numpy.vstack(data_negative)

    print(numpy.mean(data_positive, axis=0, dtype=numpy.float64))
    print(numpy.std(data_positive, axis=0, dtype=numpy.float64))
    print(numpy.mean(data_negative, axis=0, dtype=numpy.float64))
    print(numpy.std(data_negative, axis=0, dtype=numpy.float64))


    # # model training
    # event_label = 'drone'
    # model_container['models'][event_label] = {}
    # model_container['models'][event_label]['positive'] = mixture.GMM(**classifier_params).fit(data_positive)
    # model_container['models'][event_label]['negative'] = mixture.GMM(**classifier_params).fit(data_negative)

    # write
#    writeModelInfo_Txt(model_container, 'drone')
