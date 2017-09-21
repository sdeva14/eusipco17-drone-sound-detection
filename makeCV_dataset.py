import os
import numpy as np
import math
import sys

import src_dd.loadParam_manually as loadParam_manually


k_cvFold = 3


def makeCV_dataset(inputAudioPath, cvFold):

        #### create meta files as cvFold and total meta file
        # read parameter file (?.yaml) to get data path
        # params = load_parameters(parameter_file)

        params = loadParam_manually.loadParam_manual()

        data_path = params['path']['data']
        #dataDir_name = params['general']['drone_dataset']
        dataDir_name = 'Drone-sound-events-2016-development'
        local_path = os.path.join(data_path, dataDir_name)
        
        # Create the dataset path if does not exist
        if not os.path.isdir(local_path):
            os.makedirs(local_path)

        # meta.txt is total meta file, which contains information we want to experiment
        meta_filename = 'meta.txt'
        meta_file = os.path.join(local_path, meta_filename)

        fMeta_total = open(meta_file, 'w')

        metaDirPath = os.path.join(local_path, 'meta')
        sceneLabelList = os.listdir(metaDirPath)
        print sceneLabelList

        for sceneLabel in sceneLabelList:
            curPath_MetaDir = os.path.join(metaDirPath, sceneLabel)
            curMetaList = os.listdir(curPath_MetaDir)

            for curMeta in curMetaList:

                path_curMeta = os.path.join(metaDirPath, sceneLabel, curMeta)

                f_name, f_ext = os.path.splitext(curMeta)
                audio_f_name = f_name + '.wav';                
                audioDirPath = os.path.join('audio', sceneLabel, audio_f_name)

                lines = [line.rstrip('\n') for line in open(path_curMeta)]
                for line in lines:
                    subLine = line[line.find("\t")+1:].strip()
                    fMeta_total.write(audioDirPath + '\t' + sceneLabel + '\t' + subLine + '\t' + 'm' + '\n')

        fMeta_total.close()


        #### Make meta for cvFold in evaluation_setup 
        # meta information as cv fold will be stored in evaluation_setup directory
        evaluation_setup_folder = 'evaluation_setup'
        evaluation_setup_path = os.path.join(local_path, evaluation_setup_folder)
        if not os.path.isdir(evaluation_setup_path):
            os.makedirs(evaluation_setup_path)

        # # get file list for input audio path recursively
        # fileList = [];
        # for dirname, dirnames, filenames in os.walk(inputAudioPath):

        #     # print path to all subdirectories first.
        #     # for subdirname in dirnames:
        #     #     print(os.path.join(dirname, subdirname))

        #     # print path to all filenames.
        #     for filename in filenames:
        #         curFilePath = os.path.join(dirname, filename)
        #         f_name, f_ext = os.path.splitext(curFilePath)

        #         if(f_ext == '.wav'):
        #             # print(curFilePath)
        #             fileList.append(curFilePath)

        # get audio file list from total meta file
        audio_fList = [];
        lines = [line.rstrip('\n') for line in open(meta_file)]
        for line in lines:
            curFilePath = line.split("\t")[0]
            splitted_file = curFilePath.split("\\")
            curAudioFile = splitted_file[len(splitted_file)-1]
            audio_fList.append(curAudioFile)


        # Make index for train and test file as cv-fold
        numFiles = len(audio_fList)
        total_ind = np.random.permutation(numFiles)
        cvSplitted = np.array_split(total_ind, cvFold)
        
        testFile_ind = {};
        trainFile_ind = {};

        cvInd = np.arange(cvFold) + 1
        for fold in range(cvFold):
            curInd = np.delete(cvInd, fold, 0)
            testFile_ind[fold] = cvSplitted[fold]
            trainFile_ind[fold] = np.delete(cvSplitted, fold, 0)
            trainFile_ind[fold] = np.concatenate(trainFile_ind[fold],0)


        for fold in range(cvFold):
            for sceneLabel in sceneLabelList:
                # audio_fList = os.listdir(os.path.join(local_path, 'audio', sceneLabel))

                # first, write train fold
                file_train_fold = open(os.path.join(evaluation_setup_path, sceneLabel + '_' + 'fold' + str(fold+1) + '_train.txt'), 'w')
                for ind in trainFile_ind[fold]:
                    print ind, audio_fList[ind]
                    f_name, f_ext = os.path.splitext(os.path.basename(audio_fList[ind]))
                    curFMeta = f_name + '.ann'
                    curMetaPath = os.path.join(metaDirPath, sceneLabel, curFMeta)
                    lines = [line.rstrip('\n') for line in open(curMetaPath)]

                    # relPath_audio = os.path.join('audio', sceneLabel, audio_fList[ind])
                    relPath_audio = audio_fList[ind]
                    for line in lines:
                        subLine = line[line.find("\t")+1:].strip()
                        file_train_fold.write(relPath_audio + '\t' + sceneLabel + '\t' + subLine + '\n')

                file_train_fold.close()

                # then, write test fold
                file_test_fold = open(os.path.join(evaluation_setup_path, sceneLabel + '_' + 'fold' + str(fold+1) + '_test.txt'), 'w')
                file_eval_fold = open(os.path.join(evaluation_setup_path, sceneLabel + '_' + 'fold' + str(fold+1) + '_eval.txt'), 'w')
                for ind in testFile_ind[fold]:

                    f_name, f_ext = os.path.splitext(os.path.basename(audio_fList[ind]))
                    curFMeta = f_name + '.ann'
                    curMetaPath = os.path.join(metaDirPath, sceneLabel, curFMeta)
                    lines = [line.rstrip('\n') for line in open(curMetaPath)]

                    # relPath_audio = os.path.join('audio', sceneLabel, audio_fList[ind])
                    relPath_audio = audio_fList[ind]
                    for line in lines:
                        subLine = line[line.find("\t")+1:].strip()
                        print subLine
                        file_eval_fold.write(relPath_audio + '\t' + sceneLabel + '\t' + subLine + '\n')

                    file_test_fold.write(relPath_audio + '\n')

                file_test_fold.close()
                file_eval_fold.close()

    
                # second, write test fold


                # third, write evaluation fold


                # audioDirPath = os.path.join('audio', sceneLabel, audio_f_name)





def load_parameters(filename):
    """Load parameters from YAML-file

    Parameters
    ----------
    filename: str
        Path to file

    Returns
    -------
    parameters: dict
        Dict containing loaded parameters

    Raises
    -------
    IOError
        file is not found.

    """

    if os.path.isfile(filename):
        with open(filename, 'r') as f:
            return yaml.load(f)
    else:
        raise IOError("Parameter file not found [%s]" % filename)       


def splitDataset(numFiles, cvFold, total_ind, trainFile_ind, testFile_ind):
    # this function is depreciated and replaced by numpy.split_array!
    loc_cv_start = 0;
    curFileNum = numFiles;
    for fold in range(cvFold):
        nextSize = math.ceil(float(curFileNum) / (cvFold - fold))
        loc_cv_end = loc_cv_start + nextSize;

        if(loc_cv_end > numFiles):
            loc_cv_end = numFiles

        testFile_ind[fold] = total_ind[loc_cv_start : loc_cv_end]
        trainFile_ind[fold] = np.array(list(set(total_ind) - set(testFile_ind[fold])))

        loc_cv_start = loc_cv_end
        curFileNum = curFileNum - nextSize

    for i in range(len(testFile_ind)):
        print(testFile_ind[i], trainFile_ind[i])
        print('\n')


    return trainFile_ind, testFile_ind





if __name__ == "__main__":
    try:
        audioDataDir = '/home/sdeva/workspace/d_d/dd_python/data/Drone-sound-events-2016-development/audio'
        sys.exit(makeCV_dataset(audioDataDir, k_cvFold))
    except (ValueError, IOError) as e:
        sys.exit(e)