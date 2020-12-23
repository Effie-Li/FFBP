import os
from os.path import join as joinp

import ipywidgets as widgets
from IPython.display import display

import matplotlib as mpl; mpl.use('nbagg')
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from mpl_toolkits.axes_grid1 import SubplotDivider
from mpl_toolkits.axes_grid1.mpl_axes import Axes as LocatableAxes
from mpl_toolkits.axes_grid1.axes_size import Scaled

import numpy as np

from .utils import (
    load_test_data,
    list_pickles,
    get_layer_dims,
    get_data_by_key,
    get_pattern_options,
    get_layer_names,
    load_runlog
)

"""
Author: Alex Ten
Modified 12/2018 klh
Modified 12/2020 el
"""


class FigureObserver(object):
    def __init__(self, mpl_figure):
        self.fig = mpl_figure
        self.axes = mpl_figure.get_axes()
        self.widget = widgets.HTML(value='')
        self.clear_labels()
        self.fig.canvas.mpl_connect('motion_notify_event', self)

    def __call__(self, event):
        ax = event.inaxes
        if ax:
            val = ax.get_images()[0].get_cursor_data(event)
            self.update_label(val)
        else:
            self.clear_labels()

    def update_label(self, val):
        self.widget.value = '<center><samp> |{:11.5f}| </samp></center>'.format(val)

    def clear_labels(self):
        self.widget.value = '<center><samp> |cursor data| </samp></center>'


class LossDataObsever(object):
    def __init__(self, epoch_list, loss_list, loss_sum_list, tind=0, pind=0,
                 epoch_label='<center><samp> Epoch: {:4d} </samp></center>',
                 loss_label='<center><samp> Pattern loss: {:10.4} </samp></center>',
                 loss_sum_label='<center><samp> Epoch loss: {:10.4} </samp></center>'):
        self.tind, self.pind = tind, pind

        self.epoch_list = epoch_list
        self.epoch_label = epoch_label
        self.epoch_widget = widgets.HTML(value=self.epoch_label.format(epoch_list[self.tind]))

        self.loss_list = loss_list
        self.loss_label = loss_label
        self.loss_widget = widgets.HTML(value=self.loss_label.format(loss_list[self.tind][self.pind]))

        self.loss_sum_list = loss_sum_list
        self.loss_sum_label = loss_sum_label
        self.loss_sum_widget = widgets.HTML(value=self.loss_sum_label.format(loss_sum_list[self.tind]))

    def on_epoch_change(self, change):
        self.tind = change['new']
        self.epoch_widget.value = self.epoch_label.format(self.epoch_list[self.tind])
        self.loss_widget.value = self.loss_label.format(self.loss_list[self.tind][self.pind])
        self.loss_sum_widget.value = self.loss_sum_label.format(self.loss_sum_list[self.tind])

    def on_pattern_change(self, change):
        self.pind = change['new']
        self.loss_widget.value = self.loss_label.format(self.loss_list[self.tind][self.pind])

    def new_runlog(self, epoch_list, loss_list, loss_sum_list):
        self.epoch_list = epoch_list
        self.loss_list = loss_list
        self.loss_sum_list = loss_sum_list


def smooth_Gaussian(data, degree=5):
    window=degree*2-1
    weight=np.array([1.0]*window)
    weightGauss=[]

    for i in range(window):
        i=i-degree+1
        frac=i/float(window)
        gauss=1/(np.exp((4*(frac))**2))
        weightGauss.append(gauss)

    weight=np.array(weightGauss)*weight
    smoothed=[0.0]*(len(data) - window)

    for i in range(len(smoothed)):
        smoothed[i]= sum(np.array(data[i:i + window]) * weight) / sum(weight)

    return smoothed


def prog_bar(sequence, every=None, size=None, name='Items'):

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = widgets.IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = widgets.IntProgress(min=0, max=size, value=0)
    label = widgets.HTML()
    box = widgets.VBox(children=[label, progress])
    display(box)

    index = 0
    for index, record in enumerate(sequence, 1):
        if index == 1 or index % every == 0:
            if is_iterator:
                label.value = '{name}: {index} / ?'.format(
                    name=name,
                    index=index
                )
            else:
                progress.value = index
                label.value = u'{name}: {index} / {size}'.format(
                    name=name,
                    index=index,
                    size=size
                )
        yield record
    progress.bar_style = 'success'
    progress.value = index
    label.value = "{name}: {index}".format(
        name=name,
        index=str(index or '?')
    )


def _make_ghost_axis(mpl_figure, rect, title):
    ghost_ax = mpl_figure.add_axes(rect)
    [ghost_ax.spines[side].set_visible(False) for side in ['right','top','bottom','left']]
    ghost_ax.set_xticks([])
    ghost_ax.set_yticks([])
    ghost_ax.set_ylabel(title)
    ghost_ax.xaxis.set_label_coords(1, 0)
    return ghost_ax


def _make_figure(layer_dims, mode, ppc, dpi, fig_title):
    # Calculate figure size (in cell units)
    vdims, hdims = zip(*list(layer_dims.values()))
    max_width = max(hdims) + (1.8 * 4)
    fig_width = max_width + int(mode > 0) * (.8 + max_width) + int(mode > 1) * (1.8 + (max(hdims)))
    fig_height = sum(vdims) + (1.8 * len(vdims))

    # Create figure, converting cell size into inches (clip width if it exceeds 11 inches)
    fig_width, fig_height = [(dim * ppc) / dpi for dim in (fig_width, fig_height)]
    ratio = min(fig_width / fig_height, 1.5)
    fig_width = max(min(fig_width, 9), 6)
    fig_height = fig_width / ratio
    return plt.figure(num=fig_title, figsize=[fig_width, fig_height])


def _divide_axes_grid(mpl_figure, divider, layer_name, inp_size, layer_size, mode, target=False):
    '''
    DOCUMENTATION
    :param mpl_figure: an instance of matplotlib.figure.Figure
    :param N: number of layers
    :param i: layer index
    :param inp_size: number units in the layer
    :param layer_size: number of sending units to the layer
    :param target: include target
    :return:
    '''

    # provide axes-grid coordinates, image sizes, and titles
    ax_params = {
        'input_': ((0, 0), (1, inp_size), 'input'),
        'weights': ((0, 2), (layer_size, inp_size), 'W'),
        'biases': ((2, 2), (layer_size, 1), 'b'),
        'net': ((4, 2), (layer_size, 1), 'net'),
        'act': ((6, 2), (layer_size, 1), 'a')
    }
    if target: ax_params['target'] = ((8, 2), (layer_size, 1), 't')

    # define padding size
    _ = Scaled(.2)

    # define grid column sizes (left to right): weights, biases, net, act, gweight, gbiases, gnet, gact
    mat_w, cvec_w = Scaled(inp_size), Scaled(1)
    left_panel = [mat_w, _, cvec_w, _, cvec_w, _, cvec_w, _]
    cols =  left_panel + [cvec_w,_] if target else left_panel

    t = int(target)
    if mode > 0:
        right_panel = [_, mat_w, _, cvec_w, _, cvec_w, _, cvec_w]
        gax_params = {
            'gweights': ((9 + 2*t, 2), (layer_size, inp_size), 'W\''),
            'gbiases': ((11 + 2*t, 2), (layer_size, 1), 'b\''),
            'gnet': ((13 + 2*t, 2), (layer_size, 1), 'net\''),
            'gact': ((15 + 2*t, 2), (layer_size, 1), 'a\'')
        }
        for k,v in gax_params.items(): ax_params[k] = v
        if mode == 2:
            right_panel += [_, mat_w, _, cvec_w]
            ax_params['sgweights'] = ((17 + 2*t, 2), (layer_size, inp_size), 'sW\'')
            ax_params['sgbiases'] = ((19 + 2*t, 2), (layer_size, 1), 'sb\'')
        cols += right_panel

    # define grid row sizes (top to bottom): weights, input
    mat_h, rvec_h = Scaled(layer_size), Scaled(1)
    rows = [rvec_h, _, mat_h]

    # make divider
    divider.set_horizontal(cols)
    divider.set_vertical(rows)

    # create axes and locate appropriately
    img_dict = {}
    for k, (ax_coords, img_dims, ax_title) in ax_params.items():
        ax = LocatableAxes(mpl_figure, divider.get_position())
        ax.set_axes_locator(divider.new_locator(nx=ax_coords[0], ny=ax_coords[1]))
        ax.tick_params(labelbottom=False, labelleft=False)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel(ax_title) if k == 'input_' else ax.set_title(ax_title)
        if k == 'weights': ax.set_ylabel(layer_name)
        mpl_figure.add_axes(ax)
        img = ax.imshow(np.zeros(img_dims))
        img_dict[k] = img
    return img_dict


def _draw_layers(runlog_path, img_dicts, layer_names, colormap, vrange, tind, pind):
    snap = load_test_data(runlog_path=runlog_path)[tind]
    # if len(snaps) <= tind:
    #     tind = 0
    #
    # snap = snaps[tind]
    with_pind = ('input_', 'net', 'act', 'gnet', 'gact', 'gweights', 'gbiases', 'target')

    for img_dict, layer_name in zip(img_dicts, layer_names):
        for k, img in img_dict.items():

            if k == 'target':
                data = snap['target']
            else:
                data = snap[layer_name][k]

            if any([k==i for i in ('biases', 'sgbiases')]):
                data = np.expand_dims(data, axis=1)
            elif any([k == i for i in with_pind]):
                data = data[pind]
                if data.ndim < 2: data = np.expand_dims(data, axis=1)
                if any([k==i for i in ('input_', 'gweights')]): data = data.T

            img.set_data(data)
            img.cmap = get_cmap(colormap)
            img.norm.vmin = -vrange
            img.norm.vmax = vrange


def view_layers(logdir, mode=0, ppc=20):
    '''
    DOCUMENTATION
    :param logdir: path to log directory that contains pickled run logs
    :param mode: viewing mode index. Must be an int between 0 and 2
        0: limits the viewing to feedforward information only (weights, biases, net, act)
        1: same as 0, but also includes gradient information (gweights, gbiases, gnet, gact)
        2: same as 2, but also includes cumulative gradient information
    :return:
    '''
    plt.ion()
    # get runlog filenames and paths
    FILENAMES, RUNLOG_PATHS = [sorted(l) for l in list_pickles(logdir)]

    # get testing epochs and losses data
    EPOCHS, LOSSES, LOSS_SUMS = get_data_by_key(runlog_path=RUNLOG_PATHS[0], keys=['enum','loss', 'loss_sum']).values()

    # get layer names and layer dims to set up figure
    layer_names = get_layer_names(runlog_path=RUNLOG_PATHS[0])
    layer_names.reverse()
    layer_dims = get_layer_dims(runlog_path=RUNLOG_PATHS[0], layer_names=layer_names)

    # set up and make figure
    figure = _make_figure(layer_dims=layer_dims, mode=mode, ppc=ppc, dpi=96, fig_title='view_layers: '+logdir)

    num_layers = len(layer_names)
    disp_targs = [True] + [False for l in layer_names[1:]]

    axes_dicts = []
    for i, (layer_name, disp_targ) in enumerate(zip(layer_names, disp_targs)):
        sp_divider = SubplotDivider(figure, num_layers, 1, i+1, aspect=True, anchor='NW')
        vdims = [dim[0] for dim in layer_dims.values()]
        sp_divider._subplotspec._gridspec._row_height_ratios = [vdim + 1.8 for vdim in vdims]
        axes_dicts.append(
            _divide_axes_grid(
                mpl_figure=figure,
                divider = sp_divider,
                layer_name = layer_name.upper().replace('_',' '),
                inp_size = layer_dims[layer_name][1],
                layer_size = layer_dims[layer_name][0],
                mode = mode,
                target = disp_targ)
        )
    plt.tight_layout()

    _widget_layout = widgets.Layout(width='100%')

    run_widget = widgets.Dropdown(
        options=dict(zip(FILENAMES, RUNLOG_PATHS)),
        description='Run log: ',
        value=RUNLOG_PATHS[0],
        layout=_widget_layout
    )

    cmap_widget = widgets.Dropdown(
        options=sorted(['BrBG', 'bwr', 'coolwarm', 'PiYG',
                        'PRGn', 'PuOr', 'RdBu', 'RdGy',
                        'RdYlBu', 'RdYlGn', 'seismic']),
        description='Colors: ',
        value='coolwarm',
        disabled=False,
        layout = _widget_layout
    )

    vrange_widget = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=8,
        step=.1,
        description='V-range: ',
        continuous_update=False,
        layout = _widget_layout
    )

    step_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=len(EPOCHS) - 1,
        step=1,
        description='Step index: ',
        continuous_update=False,
        layout = _widget_layout
    )

    pattern_options = get_pattern_options(runlog_path=RUNLOG_PATHS[0], tind=step_widget.value)
    options_map = {}
    for i, pattern_option in enumerate(pattern_options):
        options_map[pattern_option] = i
    pattern_widget = widgets.Dropdown(
        options=options_map,
        value=0,
        description='Pattern: ',
        disabled=False,
        layout = _widget_layout
    )

    loss_observer = LossDataObsever(
        epoch_list=EPOCHS,
        loss_list=LOSSES,
        loss_sum_list=LOSS_SUMS,
        tind=step_widget.value,
        pind=pattern_widget.value,
    )

    fig_observer = FigureObserver(mpl_figure=figure)

    step_widget.observe(handler=loss_observer.on_epoch_change, names='value')
    pattern_widget.observe(handler=loss_observer.on_pattern_change, names='value')

    def on_runlog_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            newEPOCHS, newLOSSES, newLOSS_SUMS = get_data_by_key(runlog_path=change['new'],
                                                        keys=['enum', 'loss', 'loss_sum']).values()
            step_widget.max = len(newEPOCHS) - 1
            step_widget.value = 0
            pattern_widget.value = 0
            loss_observer.new_runlog(newEPOCHS, newLOSSES, newLOSS_SUMS)

    run_widget.observe(on_runlog_change)

    controls_dict = dict(
        runlog_path=run_widget,
        img_dicts=widgets.fixed(axes_dicts),
        layer_names=widgets.fixed(layer_names),
        colormap=cmap_widget,
        vrange=vrange_widget,
        tind=step_widget,
        pind=pattern_widget,
    )

    row_layout = widgets.Layout(
        display = 'flex',
        flex_flow = 'row',
        justify_content = 'center'
    )

    stretch_layout = widgets.Layout(
        display='flex',
        flex_flow='row',
        justify_content = 'space-around'
    )

    control_panel_rows = [
        widgets.Box(children=[controls_dict['runlog_path'], controls_dict['pind']], layout=row_layout),
        widgets.Box(children=[controls_dict['colormap'], controls_dict['vrange']], layout=row_layout),
        widgets.Box(children=[controls_dict['tind']], layout=row_layout),
        widgets.Box(children=[loss_observer.epoch_widget,
                              loss_observer.loss_sum_widget,
                              loss_observer.loss_widget,
                              fig_observer.widget], layout=stretch_layout)
    ]

    controls_panel = widgets.Box(
        children=control_panel_rows,
        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            padding='5px',
            border='ridge 1px',
            align_items='stretch',
            width='100%'
        )
    )

    widgets.interactive_output(f=_draw_layers, controls=controls_dict)
    display(controls_panel)

def view_layers_colab(logdir, mode=0, ppc=80, show_values=True):
    '''
    Temporary solution to get around colab not supporting interactive plot
    
    :param logdir: path to log directory that contains pickled run logs
    :param mode: viewing mode index. Must be an int between 0 and 2
        0: limits the viewing to feedforward information only (weights, biases, net, act)
        1: same as 0, but also includes gradient information (gweights, gbiases, gnet, gact)
        2: same as 2, but also includes cumulative gradient information
    :return:
    '''
    
    def _draw_static_layers(runlog_path,
                            colormap,
                            vrange,
                            tind,
                            pind,
                            show_values):
    
        # get layer names and layer dims to set up figure
        layer_names = get_layer_names(runlog_path=runlog_path)
        layer_names.reverse()
        layer_dims = get_layer_dims(runlog_path=runlog_path, layer_names=layer_names)
        
        # set up and make figure
        figure = _make_figure(layer_dims=layer_dims, mode=mode, ppc=ppc, dpi=96, fig_title='view_layers: '+logdir)

        num_layers = len(layer_names)
        disp_targs = [True] + [False for l in layer_names[1:]]
        snap = load_test_data(runlog_path=runlog_path)[tind]
        with_pind = ('input_', 'net', 'act', 'gnet', 'gact', 'gweights', 'gbiases', 'target')

        img_dicts = []
        for i, (layer_name, disp_targ) in enumerate(zip(layer_names, disp_targs)):
            sp_divider = SubplotDivider(figure, num_layers, 1, i+1, aspect=True, anchor='NW')
            vdims = [dim[0] for dim in layer_dims.values()]
            sp_divider._subplotspec._gridspec._row_height_ratios = [vdim + 1.8 for vdim in vdims]
            img_dicts.append(
                _divide_axes_grid(
                    mpl_figure=figure,
                    divider = sp_divider,
                    layer_name = layer_name.upper().replace('_',' '),
                    inp_size = layer_dims[layer_name][1],
                    layer_size = layer_dims[layer_name][0],
                    mode = mode,
                    target = disp_targ)
            )

        values_dict = {}
        for img_dict, layer_name in zip(img_dicts, layer_names):
            values_dict[layer_name] = {}
            for k, img in img_dict.items():

                if k == 'target':
                    data = snap['target']
                else:
                    data = snap[layer_name][k]

                if any([k==i for i in ('biases', 'sgbiases')]):
                    data = np.expand_dims(data, axis=1)
                elif any([k == i for i in with_pind]):
                    data = data[pind]
                    if data.ndim < 2: data = np.expand_dims(data, axis=1)
                    if any([k==i for i in ('input_',)]): data = data.T

                img.set_data(data)
                img.cmap = get_cmap(colormap)
                img.norm.vmin = -vrange
                img.norm.vmax = vrange
                if show_values:
                    for (i, j), z in np.ndenumerate(data):
                        c = 'white' if abs(z) > vrange*0.7 else 'black'
                        img.axes.text(j, i, '{:0.3f}'.format(z),
                                      size=12, color=c, ha='center', va='center')
                item_name = 'input' if k == 'input_' else img.axes.get_title()
                values_dict[layer_name][item_name] = data
    
        figure.tight_layout()
        cbax = figure.add_axes([1.02, 0.3, 0.03, 0.6]) 
        sm = plt.cm.ScalarMappable(cmap=colormap, norm=mpl.colors.Normalize(vmin=-vrange, vmax=vrange))
        plt.colorbar(sm, cbax)
        plt.show()
        
    
    # get runlog filenames and paths
    FILENAMES, RUNLOG_PATHS = [sorted(l) for l in list_pickles(logdir)]

    # get testing epochs and losses data
    EPOCHS, LOSSES, LOSS_SUMS = get_data_by_key(runlog_path=RUNLOG_PATHS[0], keys=['enum','loss', 'loss_sum']).values()

    _widget_layout = widgets.Layout(width='100%')

    run_widget = widgets.Dropdown(
        options=dict(zip(FILENAMES, RUNLOG_PATHS)),
        description='Run log: ',
        value=RUNLOG_PATHS[0],
        layout=_widget_layout
    )

    cmap_widget = widgets.Dropdown(
        options=sorted(['BrBG', 'bwr', 'coolwarm', 'PiYG',
                        'PRGn', 'PuOr', 'RdBu', 'RdGy',
                        'RdYlBu', 'RdYlGn', 'seismic']),
        description='Colors: ',
        value='coolwarm',
        disabled=False,
        layout = _widget_layout
    )

    vrange_widget = widgets.FloatSlider(
        value=1.0,
        min=0,
        max=8,
        step=.1,
        description='V-range: ',
        continuous_update=False,
        layout = _widget_layout
    )

    step_widget = widgets.IntSlider(
        value=0,
        min=0,
        max=len(EPOCHS) - 1,
        step=1,
        description='Step index: ',
        continuous_update=False,
        layout = _widget_layout
    )

    pattern_options = get_pattern_options(runlog_path=RUNLOG_PATHS[0], tind=step_widget.value)
    options_map = {}
    for i, pattern_option in enumerate(pattern_options):
        options_map[pattern_option] = i
    pattern_widget = widgets.Dropdown(
        options=options_map,
        value=0,
        description='Pattern: ',
        disabled=False,
        layout = _widget_layout
    )

    loss_observer = LossDataObsever(
        epoch_list=EPOCHS,
        loss_list=LOSSES,
        loss_sum_list=LOSS_SUMS,
        tind=step_widget.value,
        pind=pattern_widget.value,
    )

    step_widget.observe(handler=loss_observer.on_epoch_change, names='value')
    pattern_widget.observe(handler=loss_observer.on_pattern_change, names='value')

    def on_runlog_change(change):
        if change['type'] == 'change' and change['name'] == 'value':
            newEPOCHS, newLOSSES, newLOSS_SUMS = get_data_by_key(runlog_path=change['new'],
                                                        keys=['enum', 'loss', 'loss_sum']).values()
            step_widget.max = len(newEPOCHS) - 1
            step_widget.value = 0
            pattern_widget.value = 0
            loss_observer.new_runlog(newEPOCHS, newLOSSES, newLOSS_SUMS)

    run_widget.observe(on_runlog_change)

    controls_dict = dict(
        runlog_path=run_widget,
        colormap=cmap_widget,
        vrange=vrange_widget,
        tind=step_widget,
        pind=pattern_widget,
        show_values=widgets.fixed(show_values),
    )

    row_layout = widgets.Layout(
        display = 'flex',
        flex_flow = 'row',
        justify_content = 'center'
    )

    stretch_layout = widgets.Layout(
        display='flex',
        flex_flow='row',
        justify_content = 'space-around'
    )

    control_panel_rows = [
        widgets.Box(children=[controls_dict['runlog_path'], controls_dict['pind']], layout=row_layout),
        widgets.Box(children=[controls_dict['colormap'], controls_dict['vrange']], layout=row_layout),
        widgets.Box(children=[controls_dict['tind']], layout=row_layout),
        widgets.Box(children=[loss_observer.epoch_widget,
                              loss_observer.loss_sum_widget,
                              loss_observer.loss_widget], layout=stretch_layout)
    ]

    controls_panel = widgets.Box(
        children=control_panel_rows,
        layout = widgets.Layout(
            display='flex',
            flex_flow='column',
            padding='5px',
            border='ridge 1px',
            align_items='stretch',
            width='100%'
        )
    )

    plot = widgets.interactive_output(f=_draw_static_layers, controls=controls_dict)
    display(controls_panel, plot)


def view_progress(logdir, gaussian_smoothing=0, return_logs=False):
    '''
        'lr' stands for loss record
        'dGs' stands for degree of Gaussian smoothing
    '''
    plt.ion() # turn on interactive mode
    dGs = gaussian_smoothing

    # loss records and corresponding run indices will be stored lists
    lr_keys, lr_vals, lr_inds = [], [], []

    # from each runlog file inside the logdir (ending '.pkl') pull loss data and make a corresponding string index
    for runlog in [filename for filename in os.listdir(logdir) if '.pkl' in filename]:
        path = '/'.join([logdir, runlog])
        lr_keys.append('run {}'.format(runlog.split('.')[0].split('_')[1]))
        loss_log = load_runlog(path)['loss_data']
        lr_vals.append(loss_log['vals'])
        lr_inds.append(loss_log['enums'])
        if len(loss_log['vals']) <= dGs*2-1:
            msg = 'Cannot apply Gaussian smoothing with degree {}, on list of length {}. Degree must be less then (lenght+1/2)'
            raise ValueError(msg.format(dGs, len(loss_log['vals'])))

    # create new figure and axis
    fig = plt.figure(num='view_progress: ' + logdir)
    ax = fig.add_subplot(111)


    inds = [int(s.split(' ')[-1]) for s in lr_keys]
    handles = []

    for i, (key, loss_rec, rec_inds) in enumerate(zip(lr_keys, lr_vals, lr_inds)):
        if dGs:
            loss_rec = smooth_Gaussian(loss_rec, degree=dGs)
            lr_vals[i] = loss_rec
        lines, = ax.plot([i for i in range(len(loss_rec))] if dGs else rec_inds, loss_rec, alpha=.8)
        handles.append(lines)

    if len(lr_keys) > 1:
        mean_loss = np.mean(
            np.vstack([loss_rec for loss_rec in lr_vals]), axis=0
        )
        mean_line, = ax.plot([i for i in range(len(mean_loss))] if dGs else lr_inds[0], mean_loss, ls='--', lw=1.5, c='black')
        inds.append(inds[-1]+2)
        handles.append(mean_line)
        lr_keys.append('mean')
    ax.yaxis.grid()
    _ ,legend_keys, handles = zip(*sorted(zip(inds, lr_keys, handles)))

    ax.legend(handles, legend_keys, bbox_to_anchor=(1.04, 1), loc="upper left")
    ax.set_title('Training progress')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Time' if dGs else 'Epoch')
    plt.subplots_adjust(right=.8)
    if return_logs:
        return lr_keys, lr_vals, lr_inds