import os
import tensorflow as tf
import numpy as np
from collections import OrderedDict

def read_csv_file(filename_queue, batch_size, default_val, inp_size, targ_size, pattern_labels):
    reader = tf.TextLineReader(skip_header_lines=True, name='csv_reader')
    _, csv_row = reader.read_up_to(filename_queue, batch_size)
    defaults = [[default_val] for x in range(inp_size + targ_size)]
    if pattern_labels is True: 
        defaults.insert(0,[''])
    examples = tf.decode_csv(csv_row, record_defaults=defaults)
    p = tf.transpose(examples.pop(0))
    x = tf.transpose(tf.stack(examples[0:inp_size]))
    t = tf.transpose(tf.stack(examples[inp_size:inp_size+targ_size]))
    return p, x, t


def use_exercise_params(use):
    if use:
        all_vars = tf.global_variables()
        hidden_W = [v for v in all_vars if 'hidden_layer/weights' in v.name][0]
        hidden_b = [v for v in all_vars if 'hidden_layer/biases' in v.name][0]
        output_W = [v for v in all_vars if 'output_layer/weights' in v.name][0]
        output_b = [v for v in all_vars if 'output_layer/biases' in v.name][0]
        restore_dict = {'w_1': hidden_W,'b_1': hidden_b,'w_2': output_W,'b_2': output_b}
        tf.train.Saver(restore_dict).restore(tf.get_default_session(), 'exercise_params_old/exercise_params_old')


class BasicLayer(object):
    def __init__(self, layer_name, layer_input, size, wrange, nonlin=None, bias=True, seed=None, sparse_inp=False):
        self.name = layer_name
        with tf.name_scope(layer_name):
            self.input_ = layer_input
            if type(layer_input) != tf.Tensor and hasattr(layer_name, '__iter__'):
                self.input_ = tf.concat(axis=1, values=[i for i in layer_input])
            input_size = layer_input._shape[1]._value
            with tf.name_scope('weights'):

                self.weights = tf.Variable(
                    tf.random_uniform(
                        minval = wrange[0], 
                        maxval = wrange[1],
                        seed = seed,
                        shape = [input_size, size],
                        dtype=tf.float32
                    )
                )
                self.weights_summary = tf.summary.tensor_summary('params_summary', self.weights)

            self.biases = 0
            if bias:
                with tf.name_scope('biases'):
                    self.biases = tf.Variable(
                        tf.random_uniform(
                            minval = wrange[0],
                            maxval = wrange[1],
                            seed = seed,
                            shape = [1, size],
                            dtype = tf.float32
                        )
                    )
                    self.biases_summary = tf.summary.tensor_summary('params_summary', self.biases)

            with tf.name_scope('net_input'):
                self.net_input = tf.matmul(self.input_, self.weights, a_is_sparse=sparse_inp) + self.biases
                self.net_input_summary = tf.summary.tensor_summary('data_summary', self.net_input)

            with tf.name_scope('activations'):
                self.nonlin = nonlin
                if nonlin:
                    self.output = nonlin(self.net_input)
                else:
                    self.output = self.net_input
                self.output_summary = tf.summary.tensor_summary('data_summary', self.output)
    
    def add_gradient_ops(self, loss):
        with tf.name_scope(self.name):
            print(type(loss), type(self.net_input))
            self.gradient = {
                'net_input': tf.gradients(loss, self.net_input),
                'activation': tf.gradients(loss, self.output),
                'weights': tf.gradients(loss, self.weights),
                'biases': tf.gradients(loss, self.biases) if self.biases else None
            }
            self.gradient_summaries = {}
            for k, v in self.gradient.items():
                if v: self.gradient_summaries[k] = tf.summary.tensor_summary('data_summary', v)
                else: continue
     
        
class FFBP_Model(object):
    def __init__(self, layers, train_data, test_data, inp, targ, loss, optimizer):
        self.data = {'Train': train_data, 'Test': test_data}
        self.train_data = train_data
        self.test_data = test_data
        self.inp = inp
        self.targ = targ
        self.loss = loss
        tf.summary.tensor_summary('input_patterns/data_summary', self.inp)
        tf.summary.tensor_summary('target_patterns/data_summary', self.targ)
        loss_summary = tf.summary.scalar('loss_summary', self.loss)
        
        self.layers = layers
        for layer in self.layers: 
            layer.add_gradient_ops(self.loss)
        
        self.optimizer = optimizer
        self._global_step = tf.Variable(0, name='global_step', trainable=False)
        self._train_step = self.optimizer.minimize(loss=self.loss, global_step=self._global_step)
        
        self._train_fetches = {'loss':self.loss, 'summaries': tf.summary.merge([loss_summary]), '_train_step': self._train_step}
        self._test_fetches = {'loss':self.loss, 'summaries': tf.summary.merge_all()}
    
    def run_epoch(self, session, train=True, verbose=False):
        mode, fetches = ('Train', self._train_fetches) if train else ('Test', self._test_fetches)
        with tf.name_scope(mode):
            ps, xs, ts = session.run(self.data[mode])
            loss_sum = 0
            for p, x, t in zip(ps, xs, ts):
                if not train: print('label: {}\ninput: {}\ntarget: {}'.format(p, x, t)) # DEBUG (DELETE)
    #             p, x, t = session.run(self.data[mode])
                out = sess.run(
                    fetches = fetches, 
                    feed_dict = {self.inp: x, self.targ: t}
                )
                pattern_loss, summary = out[0], out[1]
                loss_sum += pattern_loss

        if verbose: 
            print('epoch {}: {}'.format(tf.train.global_step(session, self._global_step), loss_sum))
        return loss_sum, summary
    
    def save(self, path):
        pass
    
    def load(self, path):
        pass


# CONFIGS
num_epochs = 330
batch_size = 4
inp_size = 2
targ_size = 1

# QUEUES
with tf.name_scope('Train_input'):
    train_input_queue = tf.train.string_input_producer(
                    ['train_data_B.txt'], 
                    num_epochs = num_epochs, 
                    shuffle = False
    )
    
    train_examples_batch = read_csv_file(
        filename_queue = train_input_queue,
        batch_size = batch_size,
        default_val = 0.0,
        inp_size = inp_size,
        targ_size = targ_size,
        pattern_labels = True
    )

with tf.name_scope('Test_input'):
    test_input_queue = tf.train.string_input_producer(
                    ['train_data_B.txt'], 
                    num_epochs = num_epochs, 
                    shuffle = False
    )
    
    test_examples_batch = read_csv_file(
        filename_queue = test_input_queue,
        batch_size = batch_size,
        default_val = 0.0,
        inp_size = inp_size,
        targ_size = targ_size,
        pattern_labels = True
    )


# CONFIGS
# =================================================
hidden_size = 2
wrange = [-1,1]
seed = None # Use None for random seed value
lr = 0.5
m = 0.9
ckpt_freq = 1
ecrit = 0.01

# NETWORK CONSTRUCTION
with tf.name_scope('XOR_model'):
    
    model_inp  = tf.placeholder(dtype = tf.float32, shape=[None, inp_size], name='model_inp')
#     pat_labels = tf.placeholder(dtype = tf.string, shape=[batch_size, ], name='pattern_labels')
#     tf.summary.tensor_summary('pattern_labels/data_summary', pat_labels)
    
    hidden_layer = BasicLayer(
        layer_name = 'hidden_layer', 
        layer_input = model_inp, 
        size = hidden_size, 
        wrange = [-1,1], 
        nonlin=tf.nn.sigmoid, 
        bias=True, 
        seed=1, 
        sparse_inp=False
    )
    
    output_layer = BasicLayer(
        layer_name = 'output_layer', 
        layer_input = hidden_layer.output, 
        size = targ_size, 
        wrange = [-1,1], 
        nonlin=tf.nn.sigmoid, 
        bias=True, 
        seed=1, 
        sparse_inp=False
    )

    target = tf.placeholder(dtype = tf.float32, shape=[batch_size, targ_size], name='targets')
    
    xor_model = FFBP_Model(
        layers = [hidden_layer, output_layer],
        train_data = train_examples_batch, 
        test_data  = test_examples_batch,
        inp        = model_inp,
        targ       = target,
        loss       = tf.reduce_sum(tf.squared_difference(target, output_layer.output), name='loss_function'),
        optimizer  = tf.train.MomentumOptimizer(lr, m)
    )
