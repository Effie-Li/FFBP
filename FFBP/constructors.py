import os
import pickle
import tensorflow as tf
import numpy as np

from os.path import join as pjoin
from collections import OrderedDict

from .utils import new_logdir


class InputData(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, path_to_data_file, inp_size, targ_size, num_epochs, batch_size, data_len, shuffle_seed=None):
        # Store useful params
        self.path = path_to_data_file
        self.batch_size = batch_size
        self.inp_size = [inp_size] if isinstance(inp_size, int) else inp_size
        self.targ_size = targ_size
        self.data_len = data_len

        # setup filename queue
        filename_queue = tf.train.string_input_producer(string_tensor=[path_to_data_file], shuffle=False)

        # create reader and setup default values to read from files in the filename queue
        reader = tf.TextLineReader(skip_header_lines=True, name='csv_reader')
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
            num_epochs=num_epochs,
            shuffle=True if shuffle_seed else False,
            seed=shuffle_seed if shuffle_seed and shuffle_seed>=1 else None,
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

    def __init__(self, layer_name, layer_input, size, wrange, nonlin=None, bias=True, seed=None, sparse_inp=False):
        self.name = layer_name
        with tf.variable_scope(layer_name):

            if isinstance(layer_input, (list, tuple)):
                self.input_ = tf.concat(axis=1, values=[i for i in layer_input])
                input_size = sum([inp._shape[1]._value for inp in layer_input])
            else:
                self.input_ = layer_input
                input_size = layer_input._shape[1]._value

            weight_init = tf.random_uniform(
                minval=wrange[0],
                maxval=wrange[1],
                seed=seed,
                shape=[input_size, size],
                dtype=tf.float32
            )
            self.weights = tf.get_variable(name='weights', initializer=weight_init)

            self.biases = 0
            if bias:
                bias_init = tf.random_uniform(
                    minval=wrange[0],
                    maxval=wrange[1],
                    seed=seed,
                    shape=[size],
                    dtype=tf.float32
                )
                self.biases = tf.get_variable('biases', initializer=bias_init)

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
            item_keys = ['net_input', 'output', 'weights']
            items = [self.net_input, self.output, self.weights]
            if self.biases:
                item_keys.append('biases')
                items.append(self.biases)
            grad_list = tf.gradients(loss, items)
            grad_list_with_keys = [val for pair in zip(item_keys, grad_list) for val in pair]
            self.gradient = {k: v for k, v in zip(*[iter(grad_list_with_keys)] * 2)}

            for grad_op, str_key in zip(grad_list, item_keys):
                self.__dict__['g{}'.format(str_key)] = grad_op

    def fetch_test_ops(self):
        fetch_items = ['input_', 'weights', 'biases', 'net_input', 'output',
                       'gweights', 'gbiases', 'gnet_input', 'goutput']
        fetch_ops = {}
        for fi in fetch_items:
            if fi in self.__dict__.keys():
                fetch_ops[fi] = self.__dict__[fi]
        return fetch_ops, self.name


class Model(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, name, loss, optimizer, layers, inp, targ, train_data=None, test_data=None):
        self.name = name
        self.loss = loss

        self.optimizer = optimizer
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._step_incrementer = tf.assign_add(self._global_step, 1, name='global_step_incrementer')
        self._train_step = self.optimizer.minimize(loss=self.loss, global_step=None)

        self.layers = layers
        for layer in self.layers:
            layer.add_gradient_ops(loss=self.loss)

        self.inp = [inp] if not isinstance(inp, (list, tuple)) else inp
        self.targ = targ
        self.inp_labels = tf.placeholder(shape=(), dtype=tf.string)

        self.data = {'Test': test_data, 'Train': train_data}

        if train_data:
            self.data['Train'] = train_data
            self._train_fetches = {
                'loss': self.loss,
                'train_step': self._train_step,
            }

        if test_data:
            self.test_data = self.data['Test'] = test_data
            self._test_fetches = {
                'loss': self.loss,
                'enum': self._global_step,
                'labels': self.inp_labels,
                'input': tf.concat(self.inp, axis=1) if len(self.inp) > 1 else self.inp[0],
                'target': self.targ
            }

    def test_epoch(self, session, verbose=False):
        assert self.data['Test'] is not None, 'Provide test data to run a test epoch'
        data = self.data['Test']
        snap = OrderedDict()
        with tf.name_scope('Test'):
            all_examples = session.run(data.examples_batch)
            loss_sum = 0
            for example in zip(*all_examples):

                # Align lists of placeholders and feed values
                placeholders = [self.inp_labels] + self.inp + [self.targ]
                values = [example[0]] + [np.expand_dims(vec, 0) for vec in example[1:]]

                # Interleave the two lists to be for dict() constructor
                feed_list = [val for pair in zip(placeholders, values) for val in pair]

                # Construct a feed_dict with appropriately paired placeholders and feed values
                feed_dict = dict(feed_list[i:i + 2] for i in range(0, len(feed_list), 2))

                # Run graph to evaluate test fetches
                test_out = session.run(
                    fetches=self._test_fetches,
                    feed_dict=feed_dict
                )

                # Store network-level snap items: enum, loss, labels, input, target
                for k, v in test_out.items():
                    if k == 'enum':
                        snap[k] = v
                    elif k not in snap.keys():
                        snap[k] = np.expand_dims(v, axis=0)
                    else:
                        snap[k] = np.concatenate([snap[k], np.expand_dims(v, axis=0)], axis=0)

                # Store layer-level snap items: weights, biases, net_input, activations and gradients
                for layer in self.layers:
                    layer_fetches, layer_name = layer.fetch_test_ops()
                    snap.setdefault(layer_name, {})
                    layer_out = session.run(
                        fetches=layer_fetches,
                        feed_dict=feed_dict
                    )
                    for k, v in layer_out.items():
                        if k == 'weights' or k == 'biases':
                            snap[layer_name][k] = v
                        elif k not in snap[layer_name].keys():
                            snap[layer_name][k] = np.expand_dims(v, axis=0)
                        else:
                            snap[layer_name][k] = np.concatenate([snap[layer_name][k], np.expand_dims(v, axis=0)], axis=0)
                loss_sum += test_out['loss']

            # Add cumulative gradients for weights and biases
            for layer in self.layers:
                snap[layer.name]['sgweights'] = np.sum(snap[layer.name]['gweights'], axis=0)
                snap[layer.name]['sgbiases'] = np.sum(snap[layer.name]['gbiases'], axis=0)

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
                evaled_ops = session.run(
                    fetches=self._train_fetches,
                    feed_dict=feed_dict
                )
                epoch_loss += evaled_ops['loss']

        if verbose:
            print('Epoch {}: {}'.format(tf.train.global_step(session, self._global_step), epoch_loss))

        session.run(self._step_incrementer)
        return epoch_loss


class ModelSaver(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, restore_from=None, make_new_logdir=True):
        self.tf_saver = None
        self.restdir = restore_from
        self.restdir = pjoin(restore_from, 'checkpoint_files') if restore_from else None
        if make_new_logdir:
            if isinstance(make_new_logdir, (bool, int)):
                self.logdir = new_logdir()
            else:
                self.logdir = make_new_logdir
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

    def save_model(self, session, model):
        self.tf_saver = self._get_tf_saver()
        save_to = '/'.join([self.ckptdir, model.name])
        save_path = self.tf_saver.save(session, save_to, global_step=model._global_step)
        print("FFBP Saver: model saved to logdir")

    def save_test(self, snap, run_ind):
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

    def save_loss(self, loss, run_ind):
        path = '/'.join([self.logdir, 'runlog_{}.pkl'.format(run_ind)])
        try:
            with open(path, 'rb') as old_file:
                runlog = pickle.load(old_file)
            with open(path, 'wb') as old_file:
                runlog.setdefault('loss_data', []).append(loss)
                pickle.dump(runlog, old_file)
        except FileNotFoundError:
            with open(path, 'wb') as new_file:
                pickle.dump(dict(loss_data=[loss]), new_file)