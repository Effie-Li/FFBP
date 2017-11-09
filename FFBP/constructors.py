import pickle

import tensorflow as tf
import numpy as np

from FFBP.utils import new_logdir


class InputData(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, path_to_data_file, num_epochs, batch_size, inp_size, targ_size, data_len,
                 shuffle=False, shuffle_seed=None):
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
            shuffle=shuffle,
            seed=shuffle_seed,
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


class FFBPModel(object):
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

        self._prev_param = {}

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
        snap = {}
        with tf.name_scope('Test'):
            all_examples = session.run(data.examples_batch)
            loss_sum = 0
            for example in zip(*all_examples):

                # Put together lists of placeholders and values
                placeholders = [self.inp_labels] + self.inp + [self.targ]
                values = [example[0]] + [np.expand_dims(vec, 0) for vec in example[1:]]

                # Interleave the two lists to be comprehended by dict() constructor
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
                            # TODO: Include dweights and dbiases (weight change applied without the momentum term)
                            # if snap['enum'] == 0:
                            #     self._prev_param[k] = v
                            #     snap[layer_name]['d{}'.format(k)] = v*0
                            # else:
                            #     snap[layer_name]['d{}'.format(k)] =  v - self._prev_param[k]
                        elif k not in snap[layer_name].keys():
                            snap[layer_name][k] = np.expand_dims(v, axis=0)
                        else:
                            snap[layer_name][k] = np.concatenate([snap[layer_name][k], np.expand_dims(v, axis=0)],
                                                                 axis=0)
                loss_sum += test_out['loss']

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


class FFBPSaver(object):
    '''
    DOCUMENTATION
    '''

    def __init__(self, session):
        self.session = session
        self.tf_saver = tf.train.Saver(write_version=tf.train.SaverDef.V2, name='saver')
        self.logdir = None
        self.ckptdir = None

    def init_model(self, global_vars=True, local_vars=True, init_epoch=0):
        print('FFBP Saver: initializing local and global variables from scratch')
        if global_vars: self.session.run(tf.local_variables_initializer())
        if local_vars: self.session.run(tf.global_variables_initializer())
        self.logdir = new_logdir()
        self.ckptdir = self.logdir + '/checkpoint_files'
        print('FFBP Saver: new logdir at {}'.format(self.logdir))
        return init_epoch

    def restore_model(self, logdir_path, make_new_logdir=False):
        self.ckptdir = os.path.join(os.getcwd(), logdir_path, 'checkpoint_files')
        print('FFBP Saver: initializing local variables and restoring global variables from {}'.format(self.ckptdir))
        saved_files = os.listdir(self.ckptdir)
        for file in saved_files:
            if '.meta' in file:
                model_name = file.split(sep='.')[0]
                self.tf_saver.restore(self.session, os.path.join(self.ckptdir, model_name))
        self.session.run(tf.local_variables_initializer())

        if make_new_logdir:
            self.logdir = new_logdir()
            self.ckptdir = self.logdir + '/checkpoint_files'
            print('FFBP Saver: new logdir at {}'.format(self.logdir))
        else:
            self.logdir = checkpoint_dir

        restored_epoch = [self.session.run(v) for v in tf.global_variables() if 'global_step' in v.name][0]
        return restored_epoch

    def save_model(self, model):
        save_to = '/'.join([self.ckptdir, model.name])
        save_path = self.tf_saver.save(self.session, save_to, global_step=model._global_step)
        print("FFBP Saver: model saved at {}".format(save_to))

    def snap2pickle(self, snap):
        path = '/'.join([self.logdir, 'snap.pkl'])
        try:
            with open(path, 'rb') as old_file:
                old_snap = pickle.load(old_file)
            with open(path, 'wb') as old_file:
                old_snap.append(snap)
                pickle.dump(old_snap, old_file)
        except FileNotFoundError:
            with open(path, 'wb') as new_file:
                out = pickle.dump([snap], new_file)