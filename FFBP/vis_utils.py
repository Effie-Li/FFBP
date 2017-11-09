import os
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import Divider
from mpl_toolkits.axes_grid1.axes_size import Scaled

import FFBP.utils as utils
from PDPATH import PDPATH


class Logdir(object):
    def __init__(self, model_dir):
        # Create a list containing logdirs in the model directory
        self._logdirs = os.path.join(PDPATH(), model_dir, 'logdirs')
        ls = os.listdir(self._logdirs)

        # Create an interactive Dropdown widget for selecting a particular logdir and display it
        _return_ = lambda x: x
        self.widget = interactive(_return_, x = widgets.Dropdown(options=ls, description='Logs: ', value=ls[-1], disabled=False))
        display(self.widget)

        # Set the relevant paths depending on the selected logdir
        self.path = os.path.join(
            self._logdirs,
            self.widget.result
        )
        # use [x for x in os.listdir(os.path.join(self._logdirs, self.widget.result)) if '.pkl' in x].pop() in order to
        # programmatically set path based on file extension
        self.snap_path = os.path.join(
            self._logdirs,
            self.widget.result,
            'snap.pkl'
        )
        # self.event_path = os.path.join(self.log_path, os.listdir(self.log_path)[0]) TODO: identify event file by string name


def _make_axes_grid(mpl_figure, layer_size, layer_inp_size, target=False):

    # make frame to place axes inside figure
    frame = (0.1, 0.1, 0.8, 0.8)

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
    divider = Divider(fig=mpl_figure, pos=frame, horizontal=cols, vertical=rows, aspect=True)

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
        ax = mpl_figure.add_axes(frame, label=k, xticks=[], yticks=[])
        ax.set_axes_locator(divider.new_locator(nx=ax_coords[0], ny=ax_coords[1]))

        if k == 'input_':
            ax.set_xlabel(ax_title)
        else:
            ax.set_title(ax_title)
        img_dict[k] = ax.imshow(np.zeros(img_dims))
    return img_dict


def _draw_many_layers(snap_path, figures, img_dicts, layer_names, colormap, vrange, tind, pind):

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
            img.set_data(data) #, cmap=colormap, vmin=vrange[0], vmax=vrange[1])
            img.cmap = get_cmap(colormap)
            img.norm.vmin = vrange[0]
            img.norm.vmax = vrange[1]
        figures[layer_name].canvas.draw()

    plt.show()


def view_layers(logdir, layer_names, target_on_last=True):

    path = logdir.snap_path
    epochs = utils.get_epochs(snap_path=path)

    figures = {}
    axes_dicts = []

    disp_targs = [False for l in layer_names]
    disp_targs[-1] = target_on_last

    for layer_name, disp_targ in zip(layer_names, disp_targs):
        figure = plt.figure(num=layer_name.replace('_', ' '))
        size, inp_size = utils.get_layer_dims(snap_path=path, layer_name=layer_name)
        figures[layer_name] = figure
        axes_dicts.append(
            _make_axes_grid(mpl_figure=figure, layer_size=size, layer_inp_size=inp_size, target=bool(disp_targ))
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

    interact(
        _draw_many_layers,
        snap_path = fixed(logdir.snap_path),
        figures = fixed(figures),
        img_dicts = fixed(axes_dicts),
        layer_names = fixed(layer_names),
        colormap=cmap_widget,
        vrange=vrange_widget,
        tind = step_widget,
        pind = fixed(0),
    )