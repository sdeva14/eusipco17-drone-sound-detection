import src_dd.files as files

import numpy
import librosa
import scipy

#def extract_features(y, fs, mfcc_params, delta_params):
def extract_features(y, fs=25600, statistics=True, include_mfcc0=True, include_delta=True,
                       include_acceleration=True, mfcc_params=None, delta_params=None, acceleration_params=None, isMfcc=True):
    # Make feature vector
    eps = numpy.spacing(1);
    window = scipy.signal.hann(mfcc_params['n_fft'], sym=False)
    # window = scipy.signal.hamming(mfcc_params['n_fft'], sym=False)
    # window = scipy.signal.nuttall(mfcc_params['n_fft'], sym=False)

    # D = numpy.abs(librosa.stft(y))**2
    # S = librosa.feature.melspectrogram(S=D)

    # # Passing through arguments to the Mel filters
    # S = librosa.feature.melspectrogram(y=y, sr=fs, n_mels=13,
    #                                     fmax=1500)
    # print('@@')
    # print(S.shape)
    # print('!!')
    power_spectrogram = numpy.abs(librosa.stft(y + eps,
                                           n_fft=mfcc_params['n_fft'],
                                           win_length=mfcc_params['win_length'],
                                           hop_length=mfcc_params['hop_length'],
                                           center=True,
                                           window=window))**2

    # print(y.shape)
    # print(power_spectrogram.shape)

    mel_basis = librosa.filters.mel(sr=fs,
                            n_fft=mfcc_params['n_fft'],
                            n_mels=mfcc_params['n_mels'],
                            fmin=mfcc_params['fmin'],
                            fmax=mfcc_params['fmax'],
                            htk=mfcc_params['htk'])
    # print(mel_basis.shape)
    mel_spectrum = numpy.dot(mel_basis, power_spectrogram)
    # print(mel_spectrum.shape)
    mfcc = librosa.feature.mfcc(S=librosa.logamplitude(mel_spectrum),
                            n_mfcc=mfcc_params['n_mfcc'])
    # print(mfcc.shape)
    # print()

    # Collect the feature matrix
    if(isMfcc):
        feature_matrix = mfcc
 
        if include_delta:
            # Delta coefficients
            mfcc_delta = librosa.feature.delta(mfcc, **delta_params)
            # print(mfcc_delta.shape)

            # Add Delta Coefficients to feature matrix
            feature_matrix = numpy.vstack((feature_matrix, mfcc_delta))

        if not include_mfcc0:
            # Omit mfcc0
            feature_matrix = feature_matrix[1:, :]
            
        feature_matrix = feature_matrix.T;

    else:
        feature_matrix = librosa.logamplitude(mel_spectrum)
        feature_matrix = feature_matrix.T;

    # print(feature_matrix.shape)

    # Collect into data structure
    if statistics:
        return {
            'feat': feature_matrix,
            'stat': {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }
        }
    else:
        return {
            'feat': feature_matrix}

#    return feature_matrix;

def make_Normalizer(params, dataset, normalizer, ann, annDirPath):
    for item_id, audio_filename in enumerate(ann):
        audio_filepath = annDirPath[audio_filename]
        y, fs = files.load_audio(filename=dataset.relative_to_absolute_path(audio_filepath), mono=True, fs=params['features']['fs'])

        feature_data = extract_features(y=y,
                                      fs=fs,
                                      include_mfcc0=params['features']['include_mfcc0'],
                                      include_delta=params['features']['include_delta'],
                                      include_acceleration=params['features']['include_acceleration'],
                                      mfcc_params=params['features']['mfcc'],
                                      delta_params=params['features']['mfcc_delta'],
                                      acceleration_params=params['features']['mfcc_acceleration'])

        # Accumulate statistics
        normalizer.accumulate(feature_data['stat'])

    # Calculate normalization factors
    normalizer.finalize()

class FeatureNormalizer(object):
    """Feature normalizer class

    Accumulates feature statistics

    Examples
    --------

    >>> normalizer = FeatureNormalizer()
    >>> for feature_matrix in training_items:
    >>>     normalizer.accumulate(feature_matrix)
    >>>
    >>> normalizer.finalize()

    >>> for feature_matrix in test_items:
    >>>     feature_matrix_normalized = normalizer.normalize(feature_matrix)
    >>>     # used the features

    """
    def __init__(self, feature_matrix=None):
        """__init__ method.

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)] or None
            Feature matrix to be used in the initialization

        """
        if feature_matrix is None:
            self.N = 0
            self.mean = 0
            self.S1 = 0
            self.S2 = 0
            self.std = 0
        else:
            self.mean = numpy.mean(feature_matrix, axis=0)
            self.std = numpy.std(feature_matrix, axis=0)
            self.N = feature_matrix.shape[0]
            self.S1 = numpy.sum(feature_matrix, axis=0)
            self.S2 = numpy.sum(feature_matrix ** 2, axis=0)
            self.finalize()

    def __enter__(self):
        # Initialize Normalization class and return it
        self.N = 0
        self.mean = 0
        self.S1 = 0
        self.S2 = 0
        self.std = 0
        return self

    def __exit__(self, type, value, traceback):
        # Finalize accumulated calculation
        self.finalize()

    def accumulate(self, stat):
        """Accumalate statistics

        Input is statistics dict, format:

            {
                'mean': numpy.mean(feature_matrix, axis=0),
                'std': numpy.std(feature_matrix, axis=0),
                'N': feature_matrix.shape[0],
                'S1': numpy.sum(feature_matrix, axis=0),
                'S2': numpy.sum(feature_matrix ** 2, axis=0),
            }

        Parameters
        ----------
        stat : dict
            Statistics dict

        Returns
        -------
        nothing

        """
        self.N += stat['N']
        self.mean += stat['mean']
        self.S1 += stat['S1']
        self.S2 += stat['S2']

    def finalize(self):
        """Finalize statistics calculation

        Accumulated values are used to get mean and std for the seen feature data.

        Parameters
        ----------
        nothing

        Returns
        -------
        nothing

        """

        # Finalize statistics
        self.mean = self.S1 / self.N
        self.std = numpy.sqrt((self.N * self.S2 - (self.S1 * self.S1)) / (self.N * (self.N - 1)))

        # In case we have very brain-death material we get std = Nan => 0.0
        self.std = numpy.nan_to_num(self.std)

        self.mean = numpy.reshape(self.mean, [1, -1])
        self.std = numpy.reshape(self.std, [1, -1])

    def normalize(self, feature_matrix):
        """Normalize feature matrix with internal statistics of the class

        Parameters
        ----------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Feature matrix to be normalized

        Returns
        -------
        feature_matrix : numpy.ndarray [shape=(frames, number of feature values)]
            Normalized feature matrix

        """

        return (feature_matrix - self.mean) / self.std
