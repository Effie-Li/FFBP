import pickle


def snap2pickle(logdir, snap):
    path = '/'.join([logdir,'snap.pkl'])
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


def get_layer_dims(snap_path, layer_name):
    with open(snap_path, 'rb') as snap_file:
        snaps = pickle.load(snap_file)
    return snaps[0][layer_name]['weights'].shape