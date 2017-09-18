import tensorflow as tf
import functools

def lazy_property(function):
    attribute = '__cache__' + function.__name__
    
    @property
    @functools.wraps(function)
    def decorator(self):
        if not hasattr(self, attribute):
            with tf.variable_scope(function.__name):
                setattr(self, attribute, function(self))
        return getattr(self, attribute)

    return decorator

class Basic_FFBP_Model(object):
    def __init__(self, data):
        self.input_data = input_data
        self.inference
        self.optimize
        self.error
    
    @lazy_property
    def inference(self):
        
