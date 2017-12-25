# FFBP.vis_utils
# ========================

def _draw_one_layer(snap_path, axes_dict, layer_name, colormap, vrange, tind, pind):
    with open(snap_path, 'rb') as snap_file:
        snap = pickle.load(snap_file)
        enum = snap[tind]['enum']
        loss = snap[tind]['loss'][pind]
        ldict = snap[tind][layer_name]
        del snap

    print('epoch {}, loss = {:.5f}'.format(enum, loss))

    for k, ax in axes_dict.items():
        data = ldict[k]
        if k == 'weights':
            data = data.T
        if 'biases' in k:
            data = np.expand_dims(data, axis=1)
        if data.ndim > 2:
            data = data[pind]
            if k != 'input_':
                data = data.T
        ax.imshow(data, cmap=colormap, vmin=vrange[0], vmax=vrange[1])


def view_layer(logdir, layer_name, target=False):
    path = logdir.snap_path
    epochs = utils.get_epochs(snap_path=path)
    size, inp_size = utils.get_layer_dims(snap_path=path, layer_name=layer_name)

    figure = plt.figure(num=layer_name.replace('_', ' '))
    axes_dict = _make_axes_grid(mpl_figure=figure, layer_size=size, layer_inp_size=inp_size)

    cmap_widget = widgets.Dropdown(
        options=sorted(
            ['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']),
        description='Colors: ',
        value='coolwarm', disabled=False
    )

    vrange_widget = widgets.IntRangeSlider(
        value=[-1, 1],
        min=-5,
        max=5,
        step=1,
        description='V-range: ',
        continuous_update=False
    )

    step_widget = widgets.IntSlider(
        value=epochs[0],
        min=0,
        max=len(epochs) - 1,
        step=1,
        description='Step index: ',
        continuous_update=False
    )

    interact(
        _draw_one_layer,
        snap_path=fixed(logdir.snap_path),
        axes_dict=fixed(axes_dict),
        layer_name=fixed(layer_name),
        colormap=cmap_widget,
        vrange=vrange_widget,
        tind=step_widget,
        pind=fixed(3)
    )


def view_error(logdir):
    data = utils.retrieve_loss(logdir.event_path)
    plt.plot(data[:,0], data[:,1])
    plt.grid()


# import os
# from os.path import join as joinp
# from ipywidgets import interact, fixed
# import ipywidgets as widgets
#
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
# from mpl_toolkits.axes_grid1 import SubplotDivider, LocatableAxes
# from mpl_toolkits.axes_grid1.axes_size import Scaled
#
# from .utils import get_layer_dims, get_epochs, get_pattern_options


def _make_axes_grid(mpl_figure, N, subplot_ind, layer_size, layer_inp_size, target=False):
    '''
    DOCUMENTATION
    :param mpl_figure: an instance of matplotlib.figure.Figure
    :param N: number of layers
    :param i: layer index
    :param layer_size: number units in the layer
    :param layer_inp_size: number of sending units to the layer
    :param target: include target
    :return:
    '''
    # define padding size
    _ = Scaled(.8)

    # define grid column sizes (left to right): weights, biases, net_input, output, gweight, gbiases, gnet_input, goutput
    mat_w, cvec_w = Scaled(layer_size), Scaled(1)
    left_panel = [mat_w, _, cvec_w, _, cvec_w, _, cvec_w, _]
    right_panel = [_, mat_w, _, cvec_w, _, cvec_w, _, cvec_w]
    cols =  left_panel + [cvec_w,_] + right_panel if target else left_panel + right_panel

    # define grid row sizes (top to bottom): weights, input
    mat_h, rvec_h = Scaled(layer_inp_size), Scaled(1)
    rows = [rvec_h, _, mat_h]

    # make divider
    divider = SubplotDivider(mpl_figure, N, 1, subplot_ind, aspect=True)
    divider.set_horizontal(cols)
    divider.set_vertical(rows)
    # divider = Divider(fig=mpl_figure, pos=frame, horizontal=cols, vertical=rows, aspect=True)

    # provide axes-grid coordinates, image sizes, and titles
    mat_h, mat_w = layer_inp_size, layer_size
    t = int(target)
    ax_params = {
        'input_':     ((0, 0),      (1, mat_w),     'input'),
        'weights':    ((0, 2),      (mat_h, mat_w), 'W'    ),
        'biases':     ((2, 2),      (mat_h, 1),     'b'    ),
        'net_input':  ((4, 2),      (mat_h, 1),     'net'  ),
        'output':     ((6, 2),      (mat_h, 1),     'a'    ),
        'gweights':   ((9+2*t, 2),  (mat_h, mat_w), 'W\''  ),
        'gbiases':    ((11+2*t, 2), (mat_h, 1),     'b\''  ),
        'gnet_input': ((13+2*t, 2), (mat_h, 1),     'net\''),
        'goutput':    ((15+2*t, 2), (mat_h, 1),     'a\''  )
    }
    if t:
        ax_params['targets'] = ((8, 2), (mat_h, 1), 't')

    # create axes and locate appropriately
    img_dict = {}
    for k, (ax_coords, img_dims, ax_title) in ax_params.items():
        ax = LocatableAxes(mpl_figure, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=ax_coords[0], ny=ax_coords[1]))
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xticks([]); ax.set_yticks([])
        if k == 'input_':
            ax.set_xlabel(ax_title)
        else:
            ax.set_title(ax_title)
        mpl_figure.add_axes(ax)
        img_dict[k] = ax.imshow(np.zeros(img_dims))
    return img_dict


def _draw_many_layers(snap_path, img_dicts, layer_names, colormap, vrange, tind, pind):

    # pull up required data
    snap_ldicts = {}
    with open(snap_path, 'rb') as snap_file:
        snap = pickle.load(snap_file)

    enum = snap[tind]['enum']
    loss = snap[tind]['loss'][pind]
    targ = snap[tind]['target']

    for layer_name in layer_names:
        snap_ldicts[layer_name] = snap[tind][layer_name]
        snap_ldicts[layer_name]['targets'] = targ

    del snap # clean up

    print('epoch {}, loss = {:.5f}'.format(enum, loss))

    for img_dict, layer_name in zip(img_dicts, layer_names):
        for k, img in img_dict.items():
            data = snap_ldicts[layer_name][k]
            if k == 'weights':
                data = data.T
            if 'biases' in k:
                data = np.expand_dims(data, axis=1)
            if data.ndim > 2:
                data = data[pind]
                if k != 'input_':
                    data = data.T
            img.set_data(data)
            img.cmap = get_cmap(colormap)
            img.norm.vmin = vrange[0]
            img.norm.vmax = vrange[1]

    plt.show()


def view_layers(log_path, layer_names, target_on_last=True):
    filenames = [filename for filename in os.listdir(log_path) if 'log' in filename]
    runlogs = {}
    for filename in filenames:
        runlogs[filename] = joinp(log_path,filename)
    run_widget = widgets.Dropdown(
        options = runlogs,
        description = 'Run log: ',
        value = runlogs[filenames[0]]
    )


    path = run_widget.value
    epochs = get_epochs(snap_path=path)
    layer_dims = get_layer_dims(path, layer_names)

    figure = plt.figure()
    num_layers = len(layer_names)
    axes_dicts = []

    disp_targs = [False for l in layer_names]
    disp_targs[-1] = target_on_last

    for i, (layer_name, disp_targ) in enumerate(zip(layer_names, disp_targs)):
        axes_dicts.append(
            _make_axes_grid(
                mpl_figure=figure,
                N = num_layers,
                subplot_ind = i,
                layer_size=layer_dims[layer_name][0],
                layer_inp_size=layer_dims[layer_name][1],
                target=bool(disp_targ))
        )

    cmap_widget = widgets.Dropdown(
        options=sorted(['BrBG', 'bwr', 'coolwarm', 'PiYG',
                        'PRGn', 'PuOr', 'RdBu', 'RdGy',
                        'RdYlBu', 'RdYlGn', 'seismic']),
        description='Colors: ',
        value='coolwarm', disabled=False
    )

    vrange_widget = widgets.IntRangeSlider(
        value=[-1, 1],
        min=-5,
        max=5,
        step=1,
        description='V-range: ',
        continuous_update=False
    )

    step_widget = widgets.IntSlider(
        value=epochs[0],
        min=0,
        max=len(epochs) - 1,
        step=1,
        description='Step index: ',
        continuous_update=False
    )

    pattern_options = get_pattern_options(snap_path=path, tind=step_widget.value)
    options_map = {}
    for i, pattern_option in enumerate(pattern_options):
        options_map[pattern_option] = i
    pattern_widget = widgets.Select(
        options=options_map,
        value=0,
        rows=min(10, len(pattern_options)),
        description='Pattern: ',
        disabled=False
    )

    interact(
        _draw_many_layers,
        snap_path = run_widget,
        img_dicts = fixed(axes_dicts),
        layer_names = fixed(layer_names),
        colormap = cmap_widget,
        vrange = vrange_widget,
        tind = step_widget,
        pind = pattern_widget,
    )


class XY_formatter(object):
    def __init__(self, label):
        self.label = label

    def __call__(self, x, y):
        return '{0} ({2},{1}) |  '.format(self.label, int(x+.5), int(y+.5))


# FFBP.utils
# ========================
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
            if 'target_patterns' in val.tag and target: snapshot['target_pattern'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if 'test_loss_summary' in val.tag: snapshot['loss'].append(val.simple_value)

            if '/'.join([layer_name, 'weights']) in val.tag: snapshot['weights'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'biases']) in val.tag: snapshot['biases'].append(
                tf.contrib.util.make_ndarray(val.tensor))

            if '/'.join([layer_name, 'net_input']) in val.tag: snapshot['net_input'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'activation']) in val.tag: snapshot['activation'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_weights']) in val.tag: snapshot['gweights'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_biases']) in val.tag: snapshot['gbiases'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_net_input']) in val.tag: snapshot['gnet_input'].append(
                tf.contrib.util.make_ndarray(val.tensor))
            if '/'.join([layer_name, 'gradient_activation']) in val.tag: snapshot['gactivation'].append(
                tf.contrib.util.make_ndarray(val.tensor))

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
        print('> Epoch', step)
        print('  labels:')
        for x in data.labels: print(x)
        print('  inputs:')
        for x in data.inputs: print(x)
        print('  targets:')
        for x in data.targets: print(x)
        print('  data:')
        for x in data.data: print(x)


# xor
# ========================

def use_exercise_params(use):
    # DEPRICATED
    if use:
        all_vars = tf.global_variables()
        hidden_W = [v for v in all_vars if 'hidden_layer/weights' in v.name][0]
        hidden_b = [v for v in all_vars if 'hidden_layer/biases' in v.name][0]
        output_W = [v for v in all_vars if 'output_layer/weights' in v.name][0]
        output_b = [v for v in all_vars if 'output_layer/biases' in v.name][0]
        restore_dict = {'w_1': hidden_W,'b_1': hidden_b,'w_2': output_W,'b_2': output_b}
        tf.train.Saver(restore_dict, name='xor_exercise_saver').restore(
            tf.get_default_session(), 'temp/exercise_params_old/exercise_params'
        )

# FFBP.constructors
# ========================
# FOR DEBUGGING =============================
print('FEED DICT ITEMS ***'*3)
for k,v in feed_dict.items():
    print('{}:  {}'.format(k,v))

print('FETCHES ITEMS ***'*3)
for k, v in self._test_fetches.items():
    if not isinstance(v, dict):
        print('{}:  {}'.format(k, v))
    else:
        print(k.replace('_',' ').upper())
        for kk,vv in v.items():
            print('{}:  {}'.format(kk, vv))
    print()
# FOR DEBUGGING =============================