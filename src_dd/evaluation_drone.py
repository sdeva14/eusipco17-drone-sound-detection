import numpy
import math

from sklearn import metrics

class EventDetectionMetrics(object):
    """Baseclass for sound event metric classes.
    """

    def __init__(self, class_list):
        """__init__ method.

        Parameters
        ----------
        class_list : list
            List of class labels to be evaluated.

        """

        self.class_list = class_list
        self.eps = numpy.spacing(1)

    def max_event_offset(self, data):
        """Get maximum event offset from event list

        Parameters
        ----------
        data : list
            Event list, list of event dicts

        Returns
        -------
        max : float > 0
            Maximum event offset
        """

        max = 0
        for event in data:
            if event['event_offset'] > max:
                max = event['event_offset']
        return max

    def list_to_roll(self, data, time_resolution=0.01):
        """Convert event list into event roll.
        Event roll is binary matrix indicating event activity withing time segment defined by time_resolution.

        Parameters
        ----------
        data : list
            Event list, list of event dicts

        time_resolution : float > 0
            Time resolution used when converting event into event roll.

        Returns
        -------
        event_roll : numpy.ndarray [shape=(math.ceil(data_length * 1 / time_resolution), amount of classes)]
            Event roll
        """

        # Initialize
        # performance is measured for data until last offset of event, ignore remains
        data_length = self.max_event_offset(data) 
        event_roll = numpy.zeros(( int(math.ceil(data_length * 1 / time_resolution)), len(self.class_list)))

        # Fill-in event_roll
        for event in data:
            pos = self.class_list.index(event['event_label'].rstrip())

            onset = int(math.floor(event['event_onset'] * 1 / time_resolution))
            offset = int(math.ceil(event['event_offset'] * 1 / time_resolution))

            event_roll[onset:offset, pos] = 1

        return event_roll

class Metrics_Eval_Event_Drone(EventDetectionMetrics):
    def __init__(self, class_list, time_resolution=1.0, t_collar=0.2):
        """__init__ method.

        Parameters
        ----------
        class_list : list
            List of class labels to be evaluated.

        time_resolution : float > 0
            Time resolution used when converting event into event roll.
            (Default value = 1.0)

        t_collar : float > 0
            Time collar for event onset and offset condition
            (Default value = 0.2)

        """

        self.time_resolution = time_resolution
        self.t_collar = t_collar

        self.overall = {
            'Nref': 0.0,
            'Nsys': 0.0,
            'Nsubs': 0.0,
            'Ntp': 0.0,
            'Nfp': 0.0,
            'Nfn': 0.0,
        }
        self.class_wise = {}

        for class_label in class_list:
            self.class_wise[class_label] = {
                'Nref': 0.0,
                'Nsys': 0.0,
                'Ntp': 0.0,
                'Ntn': 0.0,
                'Nfp': 0.0,
                'Nfn': 0.0,
            }

        EventDetectionMetrics.__init__(self, class_list=class_list)

    def __enter__(self):
        # Initialize class and return it
        return self

    def __exit__(self, type, value, traceback):
        # Finalize evaluation and return results
        return self.results()

    def evaluate(self, annotated_ground_truth, system_output):
        """Evaluate system output and annotated ground truth pair.

        Use results method to get results.

        Parameters
        ----------
        annotated_ground_truth : numpy.array
            Ground truth array, list of scene labels

        system_output : numpy.array
            System output array, list of scene labels

        Returns
        -------
        nothing

        """

        # Overall metrics

        # Total number of detected and reference events
        Nsys = len(system_output)
        Nref = len(annotated_ground_truth)

        sys_correct = numpy.zeros(Nsys, dtype=bool)
        ref_correct = numpy.zeros(Nref, dtype=bool)

        # Number of correctly transcribed events, onset/offset within a t_collar range
        for j in range(0, len(annotated_ground_truth)):
            for i in range(0, len(system_output)):
                label_condition = annotated_ground_truth[j]['event_label'] == system_output[i]['event_label']
                onset_condition = self.onset_condition(annotated_event=annotated_ground_truth[j],
                                                       system_event=system_output[i],
                                                       t_collar=self.t_collar)

                offset_condition = self.offset_condition(annotated_event=annotated_ground_truth[j],
                                                         system_event=system_output[i],
                                                         t_collar=self.t_collar)

                if label_condition and onset_condition and offset_condition:
                    ref_correct[j] = True
                    sys_correct[i] = True
                    break

        Ntp = numpy.sum(sys_correct)

        sys_leftover = numpy.nonzero(numpy.negative(sys_correct))[0]
        ref_leftover = numpy.nonzero(numpy.negative(ref_correct))[0]

        # Substitutions
        Nsubs = 0
        for j in ref_leftover:
            for i in sys_leftover:
                onset_condition = self.onset_condition(annotated_event=annotated_ground_truth[j],
                                                       system_event=system_output[i],
                                                       t_collar=self.t_collar)

                offset_condition = self.offset_condition(annotated_event=annotated_ground_truth[j],
                                                         system_event=system_output[i],
                                                         t_collar=self.t_collar)

                if onset_condition and offset_condition:
                    Nsubs += 1
                    break

        Nfp = Nsys - Ntp - Nsubs
        Nfn = Nref - Ntp - Nsubs

        self.overall['Nref'] += Nref
        self.overall['Nsys'] += Nsys
        self.overall['Ntp'] += Ntp
        self.overall['Nsubs'] += Nsubs
        self.overall['Nfp'] += Nfp
        self.overall['Nfn'] += Nfn

        # Class-wise metrics
        for class_id, class_label in enumerate(self.class_list):
            Nref = 0.0
            Nsys = 0.0
            Ntp = 0.0

            # Count event frequencies in the ground truth
            for i in range(0, len(annotated_ground_truth)):
                if annotated_ground_truth[i]['event_label'] == class_label:
                    Nref += 1

            # Count event frequencies in the system output
            for i in range(0, len(system_output)):
                if system_output[i]['event_label'] == class_label:
                    Nsys += 1

            for j in range(0, len(annotated_ground_truth)):
                for i in range(0, len(system_output)):
                    if annotated_ground_truth[j]['event_label'] == class_label and system_output[i]['event_label'] == class_label:
                        onset_condition = self.onset_condition(annotated_event=annotated_ground_truth[j],
                                                               system_event=system_output[i],
                                                               t_collar=self.t_collar)

                        offset_condition = self.offset_condition(annotated_event=annotated_ground_truth[j],
                                                                 system_event=system_output[i],
                                                                 t_collar=self.t_collar)

                        if onset_condition and offset_condition:
                            Ntp += 1
                            break

            Nfp = Nsys - Ntp
            Nfn = Nref - Ntp

            self.class_wise[class_label]['Nref'] += Nref
            self.class_wise[class_label]['Nsys'] += Nsys

            self.class_wise[class_label]['Ntp'] += Ntp
            self.class_wise[class_label]['Nfp'] += Nfp
            self.class_wise[class_label]['Nfn'] += Nfn

    def onset_condition(self, annotated_event, system_event, t_collar=0.200):
        """Onset condition, checked does the event pair fulfill condition

        Condition:

        - event onsets are within t_collar each other

        Parameters
        ----------
        annotated_event : dict
            Event dict

        system_event : dict
            Event dict

        t_collar : float > 0
            Defines how close event onsets have to be in order to be considered match. In seconds.
            (Default value = 0.2)

        Returns
        -------
        result : bool
            Condition result

        """

        return math.fabs(annotated_event['event_onset'] - system_event['event_onset']) <= t_collar

    def offset_condition(self, annotated_event, system_event, t_collar=0.200, percentage_of_length=0.5):
        """Offset condition, checking does the event pair fulfill condition

        Condition:

        - event offsets are within t_collar each other
        or
        - system event offset is within the percentage_of_length*annotated event_length

        Parameters
        ----------
        annotated_event : dict
            Event dict

        system_event : dict
            Event dict

        t_collar : float > 0
            Defines how close event onsets have to be in order to be considered match. In seconds.
            (Default value = 0.2)

        percentage_of_length : float [0-1]


        Returns
        -------
        result : bool
            Condition result

        """
        annotated_length = annotated_event['event_offset'] - annotated_event['event_onset']
        return math.fabs(annotated_event['event_offset'] - system_event['event_offset']) <= max(t_collar, percentage_of_length * annotated_length)

    def results(self):
        """Get results

        Outputs results in dict, format:

            {
                'overall':
                    {
                        'Pre':
                        'Rec':
                        'F':
                        'ER':
                        'S':
                        'D':
                        'I':
                    }
                'class_wise':
                    {
                        'office': {
                            'Pre':
                            'Rec':
                            'F':
                            'ER':
                            'D':
                            'I':
                            'Nref':
                            'Nsys':
                            'Ntp':
                            'Nfn':
                            'Nfp':
                        },
                    }
                'class_wise_average':
                    {
                        'F':
                        'ER':
                    }
            }

        Parameters
        ----------
        nothing

        Returns
        -------
        results : dict
            Results dict

        """

        results = {
            'overall': {},
            'class_wise': {},
            'class_wise_average': {},
        }

        # Overall metrics
        results['overall']['Pre'] = self.overall['Ntp'] / (self.overall['Nsys'] + self.eps)
        results['overall']['Rec'] = self.overall['Ntp'] / self.overall['Nref']
        results['overall']['F'] = 2 * ((results['overall']['Pre'] * results['overall']['Rec']) / (results['overall']['Pre'] + results['overall']['Rec'] + self.eps))

        results['overall']['ER'] = (self.overall['Nfn'] + self.overall['Nfp'] + self.overall['Nsubs']) / self.overall['Nref']
        results['overall']['S'] = self.overall['Nsubs'] / self.overall['Nref']
        results['overall']['D'] = self.overall['Nfn'] / self.overall['Nref']
        results['overall']['I'] = self.overall['Nfp'] / self.overall['Nref']

        # Class-wise metrics
        class_wise_F = []
        class_wise_ER = []

        # print(results['overall']['Pre'])
        # print(results['overall']['Rec'])
        # print(results['overall']['F'])
        
        # print('Results')
        for class_label in self.class_list:
            if class_label not in results['class_wise']:
                results['class_wise'][class_label] = {}

            results['class_wise'][class_label]['Pre'] = self.class_wise[class_label]['Ntp'] / (self.class_wise[class_label]['Nsys'] + self.eps)
            results['class_wise'][class_label]['Rec'] = self.class_wise[class_label]['Ntp'] / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['F'] = 2 * ((results['class_wise'][class_label]['Pre'] * results['class_wise'][class_label]['Rec']) / (results['class_wise'][class_label]['Pre'] + results['class_wise'][class_label]['Rec'] + self.eps))

            results['class_wise'][class_label]['ER'] = (self.class_wise[class_label]['Nfn']+self.class_wise[class_label]['Nfp']) / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['D'] = self.class_wise[class_label]['Nfn'] / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['I'] = self.class_wise[class_label]['Nfp'] / (self.class_wise[class_label]['Nref'] + self.eps)

            results['class_wise'][class_label]['Nref'] = self.class_wise[class_label]['Nref']
            results['class_wise'][class_label]['Nsys'] = self.class_wise[class_label]['Nsys']
            results['class_wise'][class_label]['Ntp'] = self.class_wise[class_label]['Ntp']
            results['class_wise'][class_label]['Nfn'] = self.class_wise[class_label]['Nfn']
            results['class_wise'][class_label]['Nfp'] = self.class_wise[class_label]['Nfp']

            class_wise_F.append(results['class_wise'][class_label]['F'])
            class_wise_ER.append(results['class_wise'][class_label]['ER'])

            # print(class_label)
            # print(results['class_wise'][class_label]['Pre'])
            # print(results['class_wise'][class_label]['Rec'])
            # print(results['class_wise'][class_label]['F'])
            # print(results['class_wise'][class_label]['ER'])

        # Class-wise average
        results['class_wise_average']['F'] = numpy.mean(class_wise_F)
        results['class_wise_average']['ER'] = numpy.mean(class_wise_ER)

        return results

class Metrics_Eval_Segment_Drone(EventDetectionMetrics):
    """DCASE2016 Segment based metrics for sound event detection

    Supported metrics:
    - Overall
        - Error rate (ER), Substitutions (S), Insertions (I), Deletions (D)
        - F-score (F1)
    - Class-wise
        - Error rate (ER), Insertions (I), Deletions (D)
        - F-score (F1)

    Examples
    --------

    >>> overall_metrics_per_scene = {}
    >>> for scene_id, scene_label in enumerate(dataset.scene_labels):
    >>>     dcase2016_segment_based_metric = DCASE2016_EventDetection_SegmentBasedMetrics(class_list=dataset.event_labels(scene_label=scene_label))
    >>>     for fold in dataset.folds(mode=dataset_evaluation_mode):
    >>>         results = []
    >>>         result_filename = get_result_filename(fold=fold, scene_label=scene_label, path=result_path)
    >>>
    >>>         if os.path.isfile(result_filename):
    >>>             with open(result_filename, 'rt') as f:
    >>>                 for row in csv.reader(f, delimiter='\t'):
    >>>                     results.append(row)
    >>>
    >>>         for file_id, item in enumerate(dataset.test(fold,scene_label=scene_label)):
    >>>             current_file_results = []
    >>>             for result_line in results:
    >>>                 if result_line[0] == dataset.absolute_to_relative(item['file']):
    >>>                     current_file_results.append(
    >>>                         {'file': result_line[0],
    >>>                          'event_onset': float(result_line[1]),
    >>>                          'event_offset': float(result_line[2]),
    >>>                          'event_label': result_line[3]
    >>>                          }
    >>>                     )
    >>>             meta = dataset.file_meta(dataset.absolute_to_relative(item['file']))
    >>>         dcase2016_segment_based_metric.evaluate(system_output=current_file_results, annotated_ground_truth=meta)
    >>> overall_metrics_per_scene[scene_label]['segment_based_metrics'] = dcase2016_segment_based_metric.results()

    """

    def __init__(self, class_list, time_resolution=0.4):
        """__init__ method.

        Parameters
        ----------
        class_list : list
            List of class labels to be evaluated.

        time_resolution : float > 0
            Time resolution used when converting event into event roll.
            (Default value = 1.0)

        """

        self.time_resolution = time_resolution

        self.overall = {
            'Ntp': 0.0,
            'Ntn': 0.0,
            'Nfp': 0.0,
            'Nfn': 0.0,
            'Nref': 0.0,
            'Nsys': 0.0,
            'ER': 0.0,
            'S': 0.0,
            'D': 0.0,
            'I': 0.0,
        }
        self.class_wise = {}

        for class_label in class_list:
            self.class_wise[class_label] = {
                'Ntp': 0.0,
                'Ntn': 0.0,
                'Nfp': 0.0,
                'Nfn': 0.0,
                'Nref': 0.0,
                'Nsys': 0.0,
            }

        EventDetectionMetrics.__init__(self, class_list=class_list)

    def __enter__(self):
        # Initialize class and return it
        return self

    def __exit__(self, type, value, traceback):
        # Finalize evaluation and return results
        return self.results()

    def evaluate(self, system_output, annotated_ground_truth):
        """Evaluate system output and annotated ground truth pair.

        Use results method to get results.

        Parameters
        ----------
        annotated_ground_truth : numpy.array
            Ground truth array, list of scene labels

        system_output : numpy.array
            System output array, list of scene labels

        Returns
        -------
        nothing

        """

        # Convert event list into frame-based representation
        system_event_roll = self.list_to_roll(data=system_output, time_resolution=self.time_resolution)
        annotated_event_roll = self.list_to_roll(data=annotated_ground_truth, time_resolution=self.time_resolution)

        # Fix durations of both event_rolls to be equal
        if annotated_event_roll.shape[0] > system_event_roll.shape[0]:
            padding = numpy.zeros((annotated_event_roll.shape[0] - system_event_roll.shape[0], len(self.class_list)))
            system_event_roll = numpy.vstack((system_event_roll, padding))

        if system_event_roll.shape[0] > annotated_event_roll.shape[0]:
            padding = numpy.zeros((system_event_roll.shape[0] - annotated_event_roll.shape[0], len(self.class_list)))
            annotated_event_roll = numpy.vstack((annotated_event_roll, padding))

        # Compute segment-based overall metrics
        for segment_id in range(0, annotated_event_roll.shape[0]):
            annotated_segment = annotated_event_roll[segment_id, :]
            system_segment = system_event_roll[segment_id, :]

            Ntp = sum(system_segment + annotated_segment > 1)
            Ntn = sum(system_segment + annotated_segment == 0)
            Nfp = sum(system_segment - annotated_segment > 0)
            Nfn = sum(annotated_segment - system_segment > 0)

            Nref = sum(annotated_segment)
            Nsys = sum(system_segment)

            S = min(Nref, Nsys) - Ntp
            D = max(0, Nref - Nsys)
            I = max(0, Nsys - Nref)
            ER = max(Nref, Nsys) - Ntp

            self.overall['Ntp'] += Ntp
            self.overall['Ntn'] += Ntn
            self.overall['Nfp'] += Nfp
            self.overall['Nfn'] += Nfn
            self.overall['Nref'] += Nref
            self.overall['Nsys'] += Nsys
            self.overall['S'] += S
            self.overall['D'] += D
            self.overall['I'] += I
            self.overall['ER'] += ER

        # print(self.class_list)

        for class_id, class_label in enumerate(self.class_list):
            annotated_segment = annotated_event_roll[:, class_id]
            system_segment = system_event_roll[:, class_id]

            Ntp = sum(system_segment + annotated_segment > 1)
            Ntn = sum(system_segment + annotated_segment == 0)
            Nfp = sum(system_segment - annotated_segment > 0)
            Nfn = sum(annotated_segment - system_segment > 0)

            Nref = sum(annotated_segment)
            Nsys = sum(system_segment)

            self.class_wise[class_label]['Ntp'] += Ntp
            self.class_wise[class_label]['Ntn'] += Ntn
            self.class_wise[class_label]['Nfp'] += Nfp
            self.class_wise[class_label]['Nfn'] += Nfn
            self.class_wise[class_label]['Nref'] += Nref
            self.class_wise[class_label]['Nsys'] += Nsys

        return self

    def results(self):
        """Get results

        Outputs results in dict, format:

            {
                'overall':
                    {
                        'Pre':
                        'Rec':
                        'F':
                        'ER':
                        'S':
                        'D':
                        'I':
                    }
                'class_wise':
                    {
                        'office': {
                            'Pre':
                            'Rec':
                            'F':
                            'ER':
                            'D':
                            'I':
                            'Nref':
                            'Nsys':
                            'Ntp':
                            'Nfn':
                            'Nfp':
                        },
                    }
                'class_wise_average':
                    {
                        'F':
                        'ER':
                    }
            }

        Parameters
        ----------
        nothing

        Returns
        -------
        results : dict
            Results dict

        """

        results = {'overall': {},
                   'class_wise': {},
                   'class_wise_average': {},
                   }

        # Overall metrics
        # results['overall']['Pre'] = self.overall['Ntp'] / (self.overall['Nsys'] + self.eps)
        if(self.overall['Nsys'] == 0):
            results['overall']['Pre'] = 0
        else:    
            results['overall']['Pre'] = self.overall['Ntp'] / (self.overall['Nsys'])
        results['overall']['Rec'] = self.overall['Ntp'] / self.overall['Nref']
        # results['overall']['F'] = 2 * ((results['overall']['Pre'] * results['overall']['Rec']) / (results['overall']['Pre'] + results['overall']['Rec'] + self.eps))
        results['overall']['F'] = 2 * ((results['overall']['Pre'] * results['overall']['Rec']) / (results['overall']['Pre'] + results['overall']['Rec'] + self.eps))

        results['overall']['ER'] = self.overall['ER'] / self.overall['Nref']
        results['overall']['S'] = self.overall['S'] / self.overall['Nref']
        results['overall']['D'] = self.overall['D'] / self.overall['Nref']
        results['overall']['I'] = self.overall['I'] / self.overall['Nref']

        # Class-wise metrics
        class_wise_F = []
        class_wise_ER = []
        for class_id, class_label in enumerate(self.class_list):
            if class_label not in results['class_wise']:
                results['class_wise'][class_label] = {}
            results['class_wise'][class_label]['Pre'] = self.class_wise[class_label]['Ntp'] / (self.class_wise[class_label]['Nsys'] + self.eps)
            results['class_wise'][class_label]['Rec'] = self.class_wise[class_label]['Ntp'] / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['F'] = 2 * ((results['class_wise'][class_label]['Pre'] * results['class_wise'][class_label]['Rec']) / (results['class_wise'][class_label]['Pre'] + results['class_wise'][class_label]['Rec'] + self.eps))

            results['class_wise'][class_label]['ER'] = (self.class_wise[class_label]['Nfn'] + self.class_wise[class_label]['Nfp']) / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['D'] = self.class_wise[class_label]['Nfn'] / (self.class_wise[class_label]['Nref'] + self.eps)
            results['class_wise'][class_label]['I'] = self.class_wise[class_label]['Nfp'] / (self.class_wise[class_label]['Nref'] + self.eps)

            results['class_wise'][class_label]['Nref'] = self.class_wise[class_label]['Nref']
            results['class_wise'][class_label]['Nsys'] = self.class_wise[class_label]['Nsys']
            results['class_wise'][class_label]['Ntp'] = self.class_wise[class_label]['Ntp']
            results['class_wise'][class_label]['Nfn'] = self.class_wise[class_label]['Nfn']
            results['class_wise'][class_label]['Nfp'] = self.class_wise[class_label]['Nfp']

            class_wise_F.append(results['class_wise'][class_label]['F'])
            class_wise_ER.append(results['class_wise'][class_label]['ER'])

        results['class_wise_average']['F'] = numpy.mean(class_wise_F)
        results['class_wise_average']['ER'] = numpy.mean(class_wise_ER)

        return results