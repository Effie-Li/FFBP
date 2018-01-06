import os
import pickle
import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from collections import defaultdict

from .utils import new_logdir as _new_logdir


class InputData(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, path_to_data_file, inp_size, targ_size, batch_size, data_len, shuffle_seed=None):
        # Store useful params
        self.path = path_to_data_file
        self.batch_size = batch_size
        self.inp_size = [inp_size] if isinstance(inp_size, int) else inp_size
        self.targ_size = targ_size
        self.data_len = data_len

        # setup filename queue
        filename_queue = tf.train.string_input_producer(string_tensor=[path_to_data_file], shuffle=False)

        # create reader and setup default values to read from files in the filename queue
        reader = tf.TextLineReader(skip_header_lines=0, name='csv_reader')
        _, record_strings = reader.read_up_to(filename_queue, num_records=data_len)
        defaults = [[0.0] for x in range(sum(self.inp_size) + targ_size)]
        defaults.insert(0, [''])

        # decode in all lines
        examples = tf.decode_csv(record_strings, record_defaults=defaults)

        # slice the decoded lines and stack them into respective tensors
        pattern_labels = tf.transpose(examples.pop(0))
        input_patterns = []
        start = 0
        for size in self.inp_size:
            input_patterns.append(
                tf.transpose(tf.stack(examples[start:start + size]))
            )
            start += size
        target_patterns = tf.transpose(tf.stack(examples[sum(self.inp_size):sum(self.inp_size) + targ_size]))

        # enqueue lines into an examples queue (optionally shuffle)
        tensor_list = [pattern_labels] + input_patterns + [target_patterns]
        examples_slice = tf.train.slice_input_producer(
            tensor_list=tensor_list,
            shuffle=True if shuffle_seed else False,
            seed=shuffle_seed if shuffle_seed and shuffle_seed >= 1 else None,
            capacity=data_len
        )

        # set up a batch queue using the enqueued (optionally shuffled) examples
        self.examples_batch = tf.train.batch(
            tensors=examples_slice,
            batch_size=batch_size,
            capacity=batch_size
        )


class BasicLayer(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, layer_name, layer_input, size, wrange, bias=True, nonlin=None, seed=None, sparse_inp=False):
        self.name = layer_name
        with tf.variable_scope(layer_name):

            if isinstance(layer_input, (list, tuple)):
                self.input_ = tf.concat(axis=1, values=[i for i in layer_input])
                input_size = sum([inp._shape[1]._value for inp in layer_input])
            else:
                self.input_ = layer_input
                input_size = layer_input._shape[1]._value

            if seed is not None:
                seed = None if seed < 0 else seed

            weight_init = tf.random_uniform(
                minval=wrange[0],
                maxval=wrange[1],
                seed=seed,
                shape=[input_size, size],
                dtype=tf.float32
            )
            self.weights = tf.get_variable(name='weights', initializer=weight_init)

            if bias:
                bias_init = tf.random_uniform(
                    minval=wrange[0],
                    maxval=wrange[1],
                    seed=seed,
                    shape=[size],
                    dtype=tf.float32
                )
            else:
                bias_init = tf.zeros(shape=[size], dtype=tf.float32)
            self.biases = tf.get_variable('biases', initializer=bias_init, trainable=bool(bias))


        with tf.name_scope(layer_name):
            with tf.name_scope('net_input'):
                self.net_input = tf.matmul(self.input_, self.weights, a_is_sparse=sparse_inp) + self.biases

            with tf.name_scope('output'):
                self.nonlin = nonlin
                if nonlin:
                    self.output = nonlin(self.net_input)
                else:
                    self.output = self.net_input

    def add_gradient_ops(self, loss):
        with tf.name_scope(self.name):
            item_keys = ['net_input', 'output', 'weights', 'biases']
            items = [self.net_input, self.output, self.weights, self.biases]
            grad_list = tf.gradients(loss, items)
            grad_list_with_keys = [val for pair in zip(item_keys, grad_list) for val in pair]
            self.gradient = {k: v for k, v in zip(*[iter(grad_list_with_keys)] * 2)}

            for grad_op, key in zip(grad_list, item_keys):
                self.__dict__['g{}'.format(key)] = grad_op

    def fetch_test_ops(self):
        fetch_items = ['input_', 'weights', 'biases', 'net_input', 'output',
                       'gweights', 'gbiases', 'gnet_input', 'goutput']
        fetch_ops = {}
        for fi in fetch_items:
            if fi in self.__dict__.keys():
                fetch_ops[fi] = self.__dict__[fi]
        return fetch_ops


class Model(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, name, loss, layers, inp, targ, optimizer=None, train_data=None, test_data=None):
        self.name = name
        self.loss = loss
        self.layers = layers

        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._step_incrementer = tf.assign_add(self._global_step, 1, name='global_step_incrementer')
        if optimizer:
            self.set_optimizer(optimizer)
        else: self.optimizer = None

        for layer in self.layers:
            layer.add_gradient_ops(loss=self.loss)

        self.inp = [inp] if not isinstance(inp, (list, tuple)) else inp
        self.targ = targ
        self.inp_labels = tf.placeholder(dtype=tf.string, name='input_label')
        self.placeholders = [self.inp_labels] + self.inp + [self.targ]


        self.data = {'Test': test_data, 'Train': train_data}

        if train_data: self.train_setup(train_data)

        if test_data: self.test_setup(test_data)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        self._train_step = self.optimizer.minimize(loss=self.loss, global_step=None)

    def train_setup(self, data, optimizer=None):
        if optimizer: self.set_optimizer(optimizer)
        if not self.optimizer: raise ValueError('optimizer is not provided')
        self.data['Train'] = data
        self._train_fetches = {
            'loss': self.loss,
            'train_step': self._train_step,
        }

    def test_setup(self, data):
        self.test_data = self.data['Test'] = data
        self._test_fetches = {
            'enum': self._global_step,
            'loss': self.loss,
            'labels': self.inp_labels,
            'input': tf.concat(self.inp, axis=1, name='concat_input') if len(self.inp) > 1 else self.inp[0],
            'target': self.targ
        }
        for layer in self.layers:
            self._test_fetches[layer.name] = layer.fetch_test_ops()

    def test_epoch(self, session, verbose=False):
        assert self.data['Test'] is not None, 'Provide test data to run a test epoch'
        data = self.data['Test']
        assert data.batch_size == 1, 'batch_size of test data must be 1'
        snap = defaultdict(list)
        with tf.name_scope('Test'):
            loss_sum = 0
            for i in range(data.data_len // data.batch_size):

                # Make feed dict
                test_item = session.run(data.examples_batch)
                feed_dict = {}
                for placeholder, value in zip(self.placeholders, test_item):
                    feed_dict[placeholder] = value

                # Run graph to evaluate test fetches
                eval_dict = session.run(
                    fetches=self._test_fetches,
                    feed_dict=feed_dict
                )

                # Store snap items
                for K, V in eval_dict.items():
                    if not isinstance(V, dict):
                        if K == 'enum':
                            snap['enum'] = V
                        else:
                            snap[K].append(V)
                    else:
                        layer_dict = snap.setdefault(K, defaultdict(list))
                        for k, v in V.items():
                            if k == 'weights':
                                layer_dict[k] = v.T
                            elif k == 'biases':
                                layer_dict[k] = v
                            else:
                                layer_dict[k].append(v)
                loss_sum += eval_dict['loss']

            # Combine snap items into numpy array
            for K, V in snap.items():
                if K == 'enum': continue
                elif K == 'loss': snap[K] = np.array(V)
                elif isinstance(V, dict):
                    for k, v in V.items():
                        if k == 'weights' or k == 'biases': continue
                        else: snap[K][k] = np.squeeze(np.stack(v, axis=0))
                else:
                    snap[K] = np.concatenate(V, axis=0)

            # Store summary values
            for layer in self.layers:
                snap[layer.name]['sgweights'] = np.sum(snap[layer.name]['gweights'], axis=0).T
                snap[layer.name]['sgbiases'] = np.sum(snap[layer.name]['gbiases'], axis=0)
            snap['loss_sum'] = loss_sum

            if verbose:
                print('Epoch {}: {}'.format(tf.train.global_step(session, self._global_step), loss_sum))
            return loss_sum, snap

    def train_epoch(self, session, verbose=False):
        assert self.data['Train'] is not None, 'Provide train data to run a train epoch'
        data = self.data['Train']
        epoch_loss = 0
        with tf.name_scope('Train'):
            for mini_batch in range(data.data_len // data.batch_size):
                examples_batch = session.run(data.examples_batch)
                feed_list = [val for pair in zip(self.inp + [self.targ], examples_batch[1:]) for val in pair]
                feed_dict = dict(feed_list[i:i + 2] for i in range(0, len(feed_list), 2))
                eval_dict = session.run(
                    fetches=self._train_fetches,
                    feed_dict=feed_dict
                )
                epoch_loss += eval_dict['loss']

        if verbose:
            print('Epoch {}: {}'.format(tf.train.global_step(session, self._global_step), epoch_loss))
        session.run(self._step_incrementer)
        enum = session.run(self._global_step)
        return epoch_loss, enum


class ModelSaver(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, restore_from=None, logdir=None):
        self.tf_saver = None
        self.restdir = restore_from
        self.restdir = pjoin(restore_from, 'checkpoint_files') if restore_from else None
        if isinstance(logdir, str):
            try:
                os.mkdir(logdir)
                self.logdir = logdir
            except FileExistsError:
                self.logdir = logdir
        elif logdir is None:
            self.logdir = _new_logdir()
        else:
            raise ValueError('logdir arguments must be either string path or Nones ')

        self.ckptdir = pjoin(os.getcwd(), self.logdir, 'checkpoint_files')
        print('FFBP Saver: logdir path: {}'.format(self.logdir))

    def _get_tf_saver(self):
        return self.tf_saver if self.tf_saver else tf.train.Saver(write_version=tf.train.SaverDef.V2, name='saver')

    def _restore_model(self, session):
        self.tf_saver = self._get_tf_saver()
        print('FFBP Saver: initializing local variables and restoring global variables from: {}'.format(self.restdir))
        saved_files = os.listdir(self.restdir)
        saved_files.pop(saved_files.index('checkpoint'))
        model_name = saved_files[0].split(sep='.')[0]
        self.tf_saver.restore(session, pjoin(self.restdir, model_name))
        restored_epoch = [session.run(v) for v in tf.global_variables() if 'global_step' in v.name][0]
        return restored_epoch

    def init_model(self, session, init_epoch=0):
        if self.restdir:
            restored_epoch = self._restore_model(session=session)
            session.run(tf.local_variables_initializer())
            return restored_epoch
        else:
            print('FFBP Saver: initializing local and global variables from scratch')
            session.run(tf.local_variables_initializer())
            session.run(tf.global_variables_initializer())
            return init_epoch

    def save_model(self, session, model, run_ind):
        self.tf_saver = self._get_tf_saver()
        save_to = '/'.join(['{}_{}'.format(self.ckptdir, run_ind), model.name])
        save_path = self.tf_saver.save(session, save_to, global_step=model._global_step)
        print("FFBP Saver: model saved to logdir")
        return save_path

    def log_test(self, snap, run_ind):
        path = '/'.join([self.logdir, 'runlog_{}.pkl'.format(run_ind)])
        try:
            with open(path, 'rb') as old_file:
                runlog = pickle.load(old_file)
            with open(path, 'wb') as old_file:
                runlog.setdefault('test_data', []).append(snap)
                pickle.dump(runlog, old_file)
        except FileNotFoundError:
            with open(path, 'wb') as new_file:
                pickle.dump(dict(test_data=[snap]), new_file)
        return path

    def log_loss(self, loss, enum, run_ind):
        path = '/'.join([self.logdir, 'runlog_{}.pkl'.format(run_ind)])
        try:
            with open(path, 'rb') as old_file:
                runlog = pickle.load(old_file)
            with open(path, 'wb') as updated_file:
                runlog.setdefault('loss_data', {'vals':[],'enums':[]})['vals'].append(loss)
                runlog.setdefault('loss_data', {'vals':[],'enums':[]})['enums'].append(enum)
                pickle.dump(runlog, updated_file)
        except FileNotFoundError:
            with open(path, 'wb') as new_file:
                pickle.dump(dict(loss_data={'vals':[loss],'enums':[enum]}), new_file)
        return path

    def list_runlogs(self):
        return [filename for filename in os.listdir(self.logdir) if '.pkl' in filename]