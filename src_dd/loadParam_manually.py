def loadParam_manual():

	params = {}

	params['flow'] = {}
	params['flow']['initialize'] = True
	params['flow']['extract_features'] = True
	params['flow']['feature_normalizer'] = True
	params['flow']['train_system'] = True
	params['flow']['test_system'] = True
	params['flow']['evaluate_system'] = True

	params['general'] = {}
	params['general']['drone_dataset'] = 'Drone_2016_DevelopmentSet'
	# params['general']['development_dataset'] = 'TUTSoundEvents_2016_DevelopmentSet'
	# params['general']['challenge_dataset'] = 'TUTSoundEvents_2016_EvaluationSet'
	params['general']['overwrite'] = False

	params['path'] = {}
	params['path']['data'] = 'data/'
	params['path']['base'] = 'system/drone_detection/'
	params['path']['features'] = 'features/'
	params['path']['feature_normalizers'] = 'feature_normalizers/'
	params['path']['models'] = 'acoustic_models/'
	params['path']['results'] = 'evaluation_results/'

	params['features'] = {}
	params['features']['fs'] = 25600
	params['features']['win_length_seconds'] = 0.04
	params['features']['hop_length_seconds'] = 0.02
	params['features']['include_mfcc0'] = False
	params['features']['include_delta'] = False
	params['features']['include_acceleration'] = True
	params['features']['isMfcc'] = False

	params['features']['mfcc'] = {}
	params['features']['mfcc']['window'] = 'hamming_asymmetric'
	params['features']['mfcc']['n_mfcc'] = 20
	params['features']['mfcc']['n_mels'] = 40
	params['features']['mfcc']['n_fft'] = 1024
	# params['features']['mfcc']['n_fft'] = 16384
	params['features']['mfcc']['fmin'] = 0
	params['features']['mfcc']['fmax'] = 1500
	params['features']['mfcc']['htk'] = False

	params['features']['mfcc_delta'] = {}
	params['features']['mfcc_delta']['width'] = 9

	params['features']['mfcc_acceleration'] = {}
	params['features']['mfcc_acceleration']['width'] = 9

	params['classifier'] = {}
	params['classifier']['method'] = 'gmm'
	params['classifier']['parameters'] = None

	params['classifier_parameters'] = {}
	params['classifier_parameters']['gmm'] = {}
	params['classifier_parameters']['gmm']['n_components'] = 13
	params['classifier_parameters']['gmm']['covariance_type'] = 'diag'
	params['classifier_parameters']['gmm']['random_state'] = 0
	params['classifier_parameters']['gmm']['thresh'] = None
	params['classifier_parameters']['gmm']['tol'] = 0.001
	params['classifier_parameters']['gmm']['min_covar'] = 0.001
	params['classifier_parameters']['gmm']['n_iter'] = 40000
	params['classifier_parameters']['gmm']['n_init'] = 1
	params['classifier_parameters']['gmm']['params'] = 'wmc'
	params['classifier_parameters']['gmm']['init_params'] = 'kmeans'

	params['detector'] = {}
	params['detector']['decision_threshold'] = 160.0
	params['detector']['smoothing_window_length'] = 1.0		# seconds
	params['detector']['minimum_event_length'] = 0.04		# seconds
	params['detector']['minimum_event_gap'] = 0.01		# seconds


	return params