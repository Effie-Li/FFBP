import os
from os.path import join as joinp
from ipywidgets import interact, fixed
import ipywidgets as widgets

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import SubplotDivider, LocatableAxes
from mpl_toolkits.axes_grid1.axes_size import Scaled

from .utils import get_layer_dims, get_epochs, get_pattern_options, get_layer_names

def _make_logs_widget(log_path):
    filenames = [filename for filename in os.listdir(log_path) if 'runlog_' in filename]
    runlogs = {}
    for filename in filenames:
        runlogs[filename] = joinp(log_path, filename)
    run_widget = widgets.Dropdown(
        options=runlogs,
        description='Run log: ',
        value=runlogs[filenames[0]]
    )
    return run_widget


def _make_ghost_axis(mpl_figure, rect, title):
    ghost_ax = mpl_figure.add_axes(rect)
    [ghost_ax.spines[side].set_visible(False) for side in ['right','top','bottom','left']]
    ghost_ax.set_xticks([])
    ghost_ax.set_yticks([])
    ghost_ax.set_title(title)


def _make_axes_grid(mpl_figure, N, subplot_ind, layer_name, layer_size, layer_inp_size, target=False):
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

    # set suptitle
    _make_ghost_axis(mpl_figure=mpl_figure, rect=divider.get_position(), title=layer_name)

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


def _draw_layers(snap_path, img_dicts, layer_names, colormap, vrange, tind, pind):

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


def view_layers(log_path, target_on_last=True):
    run_widget = _make_logs_widget(log_path=log_path)
    path = run_widget.value
    epochs = get_epochs(log_path=path)
    layer_names = get_layer_names(log_path=path)
    layer_dims = get_layer_dims(log_path=path, layer_names=layer_names)

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
                layer_name=layer_name.upper().replace('_',' '),
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

    pattern_options = get_pattern_options(log_path=path, tind=step_widget.value)
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
        _draw_layers,
        snap_path = run_widget,
        img_dicts = fixed(axes_dicts),
        layer_names = fixed(layer_names),
        colormap = cmap_widget,
        vrange = vrange_widget,
        tind = step_widget,
        pind = pattern_widget,
    )