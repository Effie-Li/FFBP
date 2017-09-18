import tensorflow as tf

def read_csv_file(filename_queue, batch_size, default_val, inp_size, targ_size, labels):
    reader = tf.TextLineReader(skip_header_lines=True, name='csv_reader')
    _, csv_row = reader.read_up_to(filename_queue, batch_size)
    defaults = [[default_val] for x in range(inp_size + targ_size)]
    if labels is True: 
        defaults.insert(0,[''])
    examples = tf.decode_csv(csv_row, record_defaults=defaults)
    l = tf.transpose(examples.pop(0))
    x = tf.transpose(tf.stack(examples[0:inp_size]))
    t = tf.transpose(tf.stack(examples[inp_size:inp_size + targ_size]))
    return l, x, t

def xor_layer(input_tensor, input_dim, output_dim, layer_name, act=tf.nn.sigmoid):
  """Reusable code for making a simple neural net layer.
  It does a matrix multiply, bias add, and then uses a sigmoid to nonlinearize.
  It also sets up name scoping so that the resultant graph is easy to read,
  and adds a number of summary ops.
  """
  # Adding a name scope ensures logical grouping of the layers in the graph.
  with tf.name_scope(layer_name):
    # This Variable will hold the state of the weights for the layer
    with tf.name_scope('weights'):
      W = weight_variable([input_dim, output_dim])
      tf.summary.histogram('weights', W)
    with tf.name_scope('biases'):
      b = bias_variable([output_dim])
      tf.summary.histogram('biases', b)
    with tf.name_scope('net_inp'):
      net_inp = tf.nn.wx_plus_b(input_tensor, W, b)
      tf.summary.histogram('pre_activations', net_inp)
    activations = act(preactivate, name='activation')
    tf.summary.histogram('activations', activations)
    return activations