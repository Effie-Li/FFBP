import os
from IPython.display import display
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets

import mpl_toolkits.axes_grid.axes_size as Size
from mpl_toolkits.axes_grid import Divider
import matplotlib.pyplot as plt
import numpy as np

from FFBP.utils import retrieve_loss, retrieve_array
from PDPATH import PDPATH


def _return_(x):
    return x


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
    def __init__(self, path_to_logdir=PDPATH('/xor/train')):
        ls = os.listdir(path_to_logdir)
        self.__logdir_path__ = path_to_logdir
        self.widget = interactive(_return_,  x = widgets.Dropdown(options = ls, 
                                                                 description = 'Logs: ', 
                                                                 value = ls[-1], 
                                                                 disabled = False)
                                 )
        display(self.widget)
        
    def log_path(self):
        return '/'.join([self.__logdir_path__, self.widget.result])
        

def view_error(logdir):
    data = retrieve_loss(logdir.log_path())
    plt.plot(data[:,0], data[:,1])

    
def _draw_layer_(fig, data, colormap, step_ind, pat_ind, vrange):
    rect = (0.1, 0.1, 0.8, 0.8) 
    axs = [fig.add_axes(rect, label='{}'.format(i)) for i in range(len(data.keys()))]
    
    max_width, max_height = np.shape(data['weights'][step_ind])
    col_pad, row_pad = Size.Scaled(.8), Size.Scaled(.8)
    
    col_widths = [Size.Scaled(max_width), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1)]

    row_heights = [Size.Scaled(1), row_pad,
                   Size.Scaled(max_height)]

    # divide the axes rectangle into grid whose size is specified by horiz * vert
    divider = Divider(fig, rect, col_widths, row_heights, aspect=True)
    
    params = {
        'input':{'locs':(0,0), 'two_inds':True},
        'weights':{'locs':(0,2),'two_inds':False},
        'biases':{'locs':(2,2),'two_inds':False},
        'net_input':{'locs':(4,2), 'two_inds':True},
        'activations':{'locs':(6,2), 'two_inds':True}
    }
    
    axs = {}
    for i, (key, val) in enumerate(data.items()):
        axs[key] = fig.add_axes(rect, label=key, xticks=[], yticks=[])
        axs[key].set_axes_locator(divider.new_locator(nx=params[key]['locs'][0], ny=params[key]['locs'][1]))
        x = val[step_ind][pat_ind] if params[key]['two_inds'] else val[step_ind]
        axs[key].imshow(x, cmap=colormap, vmin=vrange[0], vmax=vrange[1])
    plt.draw()
    plt.show()


def view_layer(logdir, layer_name, _target=False):
    path = logdir.log_path()
    data = {'weights': None, 'net_input': None, 'biases': None, 'activations': None}
    for key in data.keys():
        data[key], steps = retrieve_array(path, layer_name, key)
        # print('retrieving {}:\n{}'.format(key, data[key]))
    print('Fake input of length:',data['weights'][0].shape[1])
    data['input'] = [np.zeros([4, data['weights'][0].shape[0]]) for i in steps]
    fig = plt.figure()

    # Axes grid is a rectangle placed inside a figure. specify the position of the rectange in the figure: 
    # (lower-left corner coords, upper-right corner coords), or (x1,y1, x2,y2)
    rect = (0.1, 0.1, 0.8, 0.8)
    max_height, max_width = np.shape(data['weights'][0])
    col_pad = Size.Scaled(.8)
    row_pad = Size.Scaled(.8)
    col_widths = [Size.Scaled(max_width), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1), col_pad,
                  Size.Scaled(1)]

    row_heights = [Size.Scaled(1), row_pad,
                   Size.Scaled(max_height)]

    # divide the axes rectangle into grid whose size is specified by horiz * vert
    Divider(fig, rect, col_widths, row_heights, aspect=True)
    
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
        value = steps[0],
        min = 0,
        max = len(steps)-1,
        step = 1,
        description='Step index: ',
        continuous_update=False)
    
    interact(
        _draw_layer_,
        fig = fixed(fig),
        data = fixed(data),
        step_ind = step_widget,
        pat_ind = fixed(0),
        colormap = cmap_widget,
        vrange=vrange_widget)
