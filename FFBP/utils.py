import os
import pickle
import numpy as np

def listify(x):
    return [x] if not isinstance(x, (list, tuple)) else x


def snap2pickle(logdir, snap, run_index):
    # Deprecated: this function is defined as ModelSaver method in FFBP.constructors
    path = '/'.join([logdir,'snap_{}.pkl'.format(run_index)])
    try:
        with open(path, 'rb') as old_file:
            old_snap = pickle.load(old_file)
        with open(path, 'wb') as old_file:
            old_snap.append(snap)
            pickle.dump(old_snap, old_file)
    except FileNotFoundError:
        with open(path, 'wb') as new_file:
            out = pickle.dump([snap], new_file)


def new_logdir():
    i=0
    logdir = os.getcwd() + '/logdirs/ffbp_logdir_000'
    while os.path.exists(logdir):
        i+=.001
        logdir = os.getcwd() + '/logdirs/ffbp_logdir_{}'.format(str(i)[2:5])
    os.makedirs(logdir)
    return logdir


def unpickle_snap(snap_path):
    with open(snap_path, 'rb') as snap_file:
        snap = pickle.load(snap_file)
    return snap


def get_epochs(snap_path):
    with open(snap_path, 'rb') as snap_file:
        snaps = pickle.load(snap_file)
    return [snap['enum'] for snap in snaps]


def get_layer_dims(snap_path, layer_names):
    layer_names = listify(layer_names)
    layer_dims = {}
    with open(snap_path, 'rb') as snap_file:
        snaps = pickle.load(snap_file)
    for layer_name in layer_names:
        layer_dims[layer_name] = snaps[0][layer_name]['weights'].shape
    return layer_dims


def get_pattern_options(snap_path, tind, input_dtype=int):
    with open(snap_path, 'rb') as snap_file:
        snap = pickle.load(snap_file)[tind]
        labels, vectors = snap['labels'], snap['input']
        del snap

    pattern_labels, pattern_vectors = [], []

    for label, vector in zip(labels, vectors):
        pattern_labels.append(label.decode('utf-8'))

        vector_string = np.array2string(vector.astype(input_dtype), separator=',', suppress_small=True)
        pattern_vectors.append(vector_string.replace('[','').replace(']',''))

    return ['{} | {}'.format(pl,pv) for pl, pv in zip(pattern_labels, pattern_vectors)]
