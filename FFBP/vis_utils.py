import os
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import AxesGrid, ImageGrid, Divider
from mpl_toolkits.axes_grid1.axes_size import Scaled

import FFBP.utils as utils
from PDPATH import PDPATH


# def _return_(x):
#     return x


def _hide_ticks(ax):
    ax.tick_params(
        axis='x',
        which='both',
        bottom='off',
        top='off',
        labelbottom='off',
        labeltop='off')
    ax.tick_params(
        axis='y',
        which='both',
        left='off',
        right='off',
        labelleft='off',
        labelright='off')


class Logdir(object):
    def __init__(self, model_dir):
        # Create a list containing logdirs in the model directory
        self._logdirs = os.path.join(model_dir, 'logdirs')
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


def view_error(logdir):
    data = utils.retrieve_loss(logdir.event_path)
    plt.plot(data[:,0], data[:,1])
    plt.grid()


def _add_grid(figure, rect, divider, x, coords, colormap, vrange):
    ax = figure.add_axes(rect, xticks=[], yticks=[])
    ax.set_axes_locator(divider.new_locator(nx=coords[0], ny=coords[1]))
    ax.imshow(x, cmap=colormap, vmin=vrange[0], vmax=vrange[1])


def _draw_layer_(fig, logdir, layer_name, colormap, epoch, pattern_label, vrange):
    # TODO !!!!!!!
    # Perhaps, draw everything upon the view_layer() call and make draw layer just insert and draw the arrays into their respective grids
    
    NUM_GRIDS = 5
    data = utils.get_layer_snapshot(logdir.event_path, layer_name, epoch, pattern_label, target=False)
    rect = (0.1, 0.1, 0.8, 0.8) 
    
    # axs = [fig.add_axes(rect, label='{}'.format(i)) for i in range(NUM_GRIDS)]
    
    max_width, max_height = 2,2
    col_pad, row_pad = Size.Scaled(.8), Size.Scaled(.8)

    col_widths = [Size.Scaled(max_width), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1)]


    row_heights = [Size.Scaled(1), row_pad, Size.Scaled(max_height)]

    divider = Divider(fig, rect, col_widths, row_heights, aspect=True)
    
    # params = {
    #     'input':{'locs':(0,0), 'two_inds':True},
    #     'weights':{'locs':(0,2),'two_inds':False},
    #     'biases':{'locs':(2,2),'two_inds':False},
    #     'net_input':{'locs':(4,2), 'two_inds':True},
    #     'activations':{'locs':(6,2), 'two_inds':True}
    # }
    # 
    # axs = {}
    # for i, (key, val) in enumerate(data.items()):
    #     axs[key] = fig.add_axes(rect, label=key, xticks=[], yticks=[])
    #     axs[key].set_axes_locator(divider.new_locator(nx=params[key]['locs'][0], ny=params[key]['locs'][1]))
    #     x = val[step_ind][pat_ind] if params[key]['two_inds'] else val[step_ind]
    #     axs[key].imshow(x, cmap=colormap, vmin=vrange[0], vmax=vrange[1])
    
    _add_grid(fig, rect, divider, data['input_pattern'], (0,0), colormap, vrange)
    _add_grid(fig, rect, divider, data['weights'], (0,2), colormap, vrange)
    _add_grid(fig, rect, divider, data['biases'], (2,2), colormap, vrange)
    _add_grid(fig, rect, divider, data['net_input'], (4,2), colormap, vrange)
    _add_grid(fig, rect, divider, data['activation'], (6,2), colormap, vrange)

    
    plt.draw()
    plt.show()


def view_layer(logdir, layer_name, target=False):

    path = logdir.snap_path
    epochs = utils.get_epochs(snap_path=path)
    inp_size, size = utils.get_layer_dims(snap_path=path, layer_name=layer_name)

    figure = plt.figure(1)

    ax_dict = make_axes_grid(mpl_figure=figure, layer_size=size, layer_inp_size=inp_size)

    cmap_widget = widgets.Dropdown(
        options = sorted(['BrBG', 'bwr', 'coolwarm', 'PiYG', 'PRGn', 'PuOr', 'RdBu', 'RdGy', 'RdYlBu', 'RdYlGn', 'seismic']),  
        description = 'Colors: ', 
        value = 'RdBu',  disabled = False)
    
    vrange_widget = widgets.IntRangeSlider(
        value=[-1, 1],
        min=-5,
        max=5,
        step=1,
        description='V-range: ',
        continuous_update=False)
    
    step_widget = widgets.IntSlider(
        value = epochs[0],
        min = 0,
        max = len(epochs)-1,
        step = 1,
        description='Step index: ',
        continuous_update=False)
    
    # interact(
    #     _draw_layer_,
    #     fig = fixed(fig),
    #     logdir = fixed(logdir),
    #     layer_name = fixed(layer_name),
    #     epoch = step_widget,
    #     pattern_label = fixed('p00'),
    #     colormap = cmap_widget,
    #     vrange=vrange_widget)


def make_axes_grid(mpl_figure, layer_size, layer_inp_size, target=False):

    # make frame
    frame = (0.1, 0.1, 0.9, 0.9)

    # define padding size
    _ = Scaled(.8)

    # grid column widths (left to right): weights, biases, net_input, output, gweight, gbiases, gnet_input, goutput
    w, b, n, o = Scaled(layer_size), Scaled(1), Scaled(1), Scaled(1)
    gw, gb, gn, go = Scaled(layer_size), Scaled(1), Scaled(1), Scaled(1)
    left_panel = [w, _, b, _, n, _, o, _]
    right_panel = [_, gw, _, gb, _, gn, _, go]
    if target:
        left_panel += [Scaled(1),_]
    col_sizes = left_panel + right_panel

    # row sizes (top to bottom): weights, input
    w, i = Scaled(layer_inp_size), Scaled(1)
    row_sizes = [w, _, i]

    # make divider
    divider = Divider(fig=mpl_figure, pos=frame, vertical=col_sizes, horizontal=row_sizes, aspect=True)

    # provide coords for axes for data arrays
    _t_ = int(target)*2
    coords = {
        'weights': (0, 0),
        'biases': (0, 2),
        'net_input': (0, 4),
        'output': (0, 6),
        'gweights': (0, 9 + _t_),
        'gbiases': (0, 11 + _t_),
        'gnet_input': (0, 13 + _t_),
        'goutput': (0, 15 + _t_),
        'layer_input': (1, 0)
    }
    if target: coords['targ'] = (0, 8)

    axes_dict = {}
    for ax_key, ax_coords in coords.items():
        ax = mpl_figure.add_axes(frame, xticks=[], yticks=[])
        ax.set_axes_locator(divider.new_locator(nx=ax_coords[0], ny=ax_coords[1]))
        axes_dict[ax_key] = ax

    return axes_dict