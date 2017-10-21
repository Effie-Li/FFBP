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


def get_layer_snapshot(path_to_event_file, layer_name, epoch, pattern_label, target=False):
    '''
    
    :param path_to_event_file: 
    :param epoch: 
    :param pattern_label: 
    :param target: 
    :return: snapshot = dict(input_pattern=[], weights=[], biases=[], net_input=[], activation=[],
                    gweights=[], gbiases=[], gnet_input=[], gactivation=[], loss=[])
    '''
    snapshot = dict(input_pattern=[], weights=[], biases=[], net_input=[], activation=[],
                    gweights=[], gbiases=[], gnet_input=[], gactivation=[], loss=[])
    if target: fields.append('target')
    
    events = []
    for event in tf.train.summary_iterator(path_to_event_file):
        if event.step == epoch:
            events.append(event)
    
    labels = []
    for event in events:
        for val in event.summary.value:
            if 'pattern_labels' in val.tag: labels.append(val.tensor.string_val[0].decode('utf-8'))
            if 'input_patterns' in val.tag: snapshot['input_pattern'].append(tf.contrib.util.make_ndarray(val.tensor))
            if 'target_patterns' in val.tag and target: snapshot['target_pattern'].append(tf.contrib.util.make_ndarray(val.tensor))
            if 'test_loss_summary' in val.tag: snapshot['loss'].append(val.simple_value)
                
            if '/'.join([layer_name,'weights']) in val.tag: snapshot['weights'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'biases']) in val.tag: snapshot['biases'].append(tf.contrib.util.make_ndarray(val.tensor))
    
            if '/'.join([layer_name, 'net_input']) in val.tag: snapshot['net_input'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'activation']) in val.tag: snapshot['activation'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_weights']) in val.tag: snapshot['gweights'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_biases']) in val.tag: snapshot['gbiases'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_net_input']) in val.tag: snapshot['gnet_input'].append(tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_activation']) in val.tag: snapshot['gactivation'].append(tf.contrib.util.make_ndarray(val.tensor))
    
    ind = labels.index(pattern_label)
    for key, snap_val in snapshot.items():
        snapshot[key] = snap_val[ind]
        
    return snapshot


def get_epochs_and_labels(path_to_event_file):
    epochs = []
    labels = []
    got_labels = False
    for event in tf.train.summary_iterator(path_to_event_file):
        for val in event.summary.value:
            if 'test_loss_summary' in val.tag: epochs.append(event.step)
        if not got_labels:
            for val in event.summary.value:
                if 'pattern_labels' in val.tag:
                    labels.append(val.tensor.string_val[0].decode('utf-8'))
            got_labels = True
    return list(set(epochs)), labels


def retrieve_model_data(path_to_event_file, layer_name, tensor_name):
    '''
    Retrieves model data from a tensorflow event file and structures it into lists within a named tuple withina an 
    ordered dict
    :param path_to_event_file: absolute path to event file
    :param layer_name: string name of the layer
    :param tensor_name: string name of the data to retrieve (e.g. 'net_input')
    :return data_dict: 
    example: OrderedDict = {epoch_num: namedtuple(labels: [t1,t2], inputs: [t1,t2], targets: [t1,t2], data: [t1,t2])}
    '''
    data_dict = OrderedDict()
    SummaryData = namedtuple('SummaryData', ['labels', 'inputs', 'targets', 'data'])
    lookup = '/'.join([layer_name, tensor_name])
    for event in tf.train.summary_iterator(path_to_event_file):
        data_dict.setdefault(event.step, SummaryData([], [], [], []))
        for val in event.summary.value:
            if 'pattern_labels' in val.tag:
                data_dict[event.step].labels.append(val.tensor.string_val[0].decode('utf-8'))
            elif 'input_patterns' in val.tag:
                data_dict[event.step].inputs.append(tf.contrib.util.make_ndarray(val.tensor))
            elif 'target_patterns' in val.tag:
                data_dict[event.step].targets.append(tf.contrib.util.make_ndarray(val.tensor))
            elif lookup in val.tag:
                data_dict[event.step].data.append(tf.contrib.util.make_ndarray(val.tensor))
        if not sum([len(field) for field in data_dict[event.step]]): data_dict.pop(event.step)
    return data_dict


def retrieve_loss(path_to_event_file):
    lookup = 'train_loss_summary'
    epochs, loss_vals = [], []
    
    for event in tf.train.summary_iterator(path_to_event_file):
        for val in event.summary.value:
            if lookup in val.tag:
                epochs.append(event.step)
                loss_vals.append(val.simple_value)
                
    return np.column_stack([epochs, loss_vals])


def display_model_data(data_dict):
    for step, data in data_dict.items():
        print('> Epoch',step)
        print('  labels:')
        for x in data.labels: print(x)
        print('  inputs:')
        for x in data.inputs: print(x)
        print('  targets:')
        for x in data.targets: print(x)
        print('  data:')
        for x in data.data: print(x)
