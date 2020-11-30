from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras import backend as K
from tensorflow.python.ops import math_ops
import tensorflow as tf

class ChannelReversal(Layer):
    """Image color channel reversal layer (e.g. RGB -> BGR)."""

    def __init__(self):
        super(ChannelReversal, self).__init__()

    def call(self, inputs):
        return tf.reverse(inputs, axis=tf.constant([3]), name="channel_reversal")


class DepthwiseNormalization(Layer):
    """Channel specific normalisation"""

    def __init__(self, mean=[0.,0.,0.], stddev=[1.,1.,1.]):
        super(DepthwiseNormalization, self).__init__()
        self.mean = tf.broadcast_to(mean, [224,224,3])
        self.stddev = tf.broadcast_to(stddev, [224,224,3])   

    def call(self, inputs):
        if inputs.dtype != K.floatx():
            inputs = math_ops.cast(inputs, K.floatx())

        return (inputs - self.mean) / self.stddev

