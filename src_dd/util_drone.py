import os
import hashlib
import json

# write raw audio data as txt file with numbers
def writeAudioData(outputName, y_audio, fs):
    outF = open(outputName, 'w')

    outF.write(str(fs) + "\n")
    for i in range(len(y_audio)):
        outF.write(str(y_audio[i]) + "\n")
    # outF.write("\n")

    outF.close()

# write mfcc data as txt file with numbers
def writeMfccData(outputName, featured, dim, len):
    outF = open(outputName, 'w')

    outF.write(str(len) + " " + str(dim) + "\n")
    for i in range(len):
        for j in range(dim):
            outF.write(str(featured[i, j]) + " ")
        outF.write("\n")

    outF.close()    


# write model info (weights, means, covars) to pass to other library or implementation
def writeModelToTxt(outputName, posModel, negModel):

    # Write model to file to be loaded from other implementation
    # example)
    # outputName = "gmm_drone.model";
    # writeModelToTxt(outputName, model_container['models'][event_label]['positive'], model_container['models'][event_label]['negative'])

    outF = open(outputName, 'w')

    # writing order is weights, means, covars
    num_comp = posModel.weights_.shape[0]
    outF.write(str(num_comp) + "\n");

    for i in range(len(posModel.weights_)):
        outF.write(str(posModel.weights_[i]) + " ")
    outF.write("\n")

    for i in range(posModel.means_.shape[0]):
        for j in range(posModel.means_.shape[1]):
            outF.write(str(posModel.means_[i, j]) + " ")
        outF.write("\n");

    for i in range(posModel.covars_.shape[0]):
        for j in range(posModel.covars_.shape[1]):
            outF.write(str(posModel.covars_[i, j]) + " ")
        outF.write("\n");

    for i in range(len(negModel.weights_)):
        outF.write(str(negModel.weights_[i]) + " ")
    outF.write("\n")

    for i in range(negModel.means_.shape[0]):
        for j in range(negModel.means_.shape[1]):
            outF.write(str(negModel.means_[i, j]) + " ")
        outF.write("\n");

    for i in range(negModel.covars_.shape[0]):
        for j in range(negModel.covars_.shape[1]):
            outF.write(str(negModel.covars_[i, j]) + " ")
        outF.write("\n");

    outF.close()

def check_path(path):
    """Check if path exists, if not creates one

    Parameters
    ----------
    path : str
        Path to be checked.

    Returns
    -------
    Nothing

    """

    if not os.path.isdir(path):
        os.makedirs(path)


def get_parameter_hash(params):
    """Get unique hash string (md5) for given parameter dict

    Parameters
    ----------
    params : dict
        Input parameters

    Returns
    -------
    md5_hash : str
        Unique hash for parameter dict

    """

    md5 = hashlib.md5()
    md5.update(str(json.dumps(params, sort_keys=True)))
    return md5.hexdigest()

