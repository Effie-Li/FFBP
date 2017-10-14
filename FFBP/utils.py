from collections import OrderedDict, namedtuple

import tensorflow as tf
import numpy as np


def retrieve_model_params(path_to_event_file, layer_name, param_name):
    """
    Retrieves parameter values from a tensorflow event file and structures it into an ordered dict
    :param path_to_event_file: absolute path to event file
    :param layer_name: string name of the layer
    :param params_summary: stirng name of the parameter to retrieve (e.g. 'weights')
    :return params_dict: an ordered dict where keys are global steps (of type int) and values are numpy arrays
    """
    params_dict = OrderedDict()
    lookup = '/'.join([layer_name, param_name])
    for event in tf.train.summary_iterator(path_to_event_file):
        for val in event.summary.value:
            if lookup in val.tag:
                params_dict[event.step] = tf.contrib.util.make_ndarray(val.tensor)
    return params_dict


def retrieve_model_data(path_to_event_file, layer_name, tensor_name):
    '''
    Retrieves model data from a tensorflow event file and structures it into lists within a named tuple withina an 
    ordered dict: 
    :param path_to_event_file: absolute path to event file
    :param layer_name: string name of the layer
    :param tensor_name: string name of the data to retrieve (e.g. 'net_input')
    :return data_dict: OrderedDict = {epoch_num: namedtuple(labels: [], inputs: [], targets: [], data: [])}
    '''
    SummaryData = namedtuple('SummaryData', ['labels', 'inputs', 'targets', 'data'])
    data_dict = OrderedDict()
    lookup = '/'.join([layer_name, tensor_name])
    for i, event in enumerate(tf.train.summary_iterator(event_file)):
        data_dict.setdefault(event.step, SummaryData([],[],[],[]))
        for val in event.summary.value:
            if 'input_patterns' in val.tag:
                data_dict[event.step].inputs.append(tf.contrib.util.make_ndarray(val.tensor))
            if 'target_patterns' in val.tag:
                data_dict[event.step].targets.append(tf.contrib.util.make_ndarray(val.tensor))
            if lookup in val.tag:
                data_dict[event.step].data.append(tf.contrib.util.make_ndarray(val.tensor))

    return data_dict


def retrieve_loss(path_to_event_file):
    eacc = get_event_accumulator(path_to_event_file)
    loss_log = np.stack([np.asarray([scalar.step, scalar.value]) for scalar in eacc.Scalars('train/error_summary')])
    return loss_log


def display_model_data(data_dict):
    for step, data in data_dict.items():
        print('> Epoch',step)
        print('  inputs:')
        for x in data.inputs: print(x)
        print('  targets:')
        for x in data.targets: print(x)
        print('  data:')
        for x in data.data: print(x)
