'''VGGFace models for Keras.

# Notes:
- Resnet50 and VGG16  are modified architectures from Keras Application folder. [Keras](https://keras.io)

- Squeeze and excitation block is taken from  [Squeeze and Excitation Networks in
 Keras](https://github.com/titu1994/keras-squeeze-excite-network) and modified.

'''


from tensorflow.keras.layers import Flatten, Dense, Input, GlobalAveragePooling2D, \
    GlobalMaxPooling2D, Activation, Conv2D, MaxPooling2D, BatchNormalization, \
    AveragePooling2D, Reshape, Permute, multiply
from tensorflow.keras.utils import get_file
from tensorflow.keras import backend as K
from keras_vggface import utils
import warnings
from tensorflow.keras.models import Model
from tensorflow.keras import layers, Sequential
from tensorflow.keras.layers.experimental.preprocessing import Resizing
from .preprocessing import DepthwiseNormalization, ChannelReversal
import tensorflow as tf

def senet_se_block(input_tensor, stage, block, compress_rate=16, bias=False):
    conv1_down_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_down"
    conv1_up_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_up"

    num_channels = int(input_tensor.shape[-1])
    bottle_neck = int(num_channels // compress_rate)

    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape((1, 1, num_channels))(se)
    se = Conv2D(bottle_neck, (1, 1), use_bias=bias,
                name=conv1_down_name)(se)
    se = Activation('relu')(se)
    se = Conv2D(num_channels, (1, 1), use_bias=bias,
                name=conv1_up_name)(se)
    se = Activation('sigmoid')(se)

    x = input_tensor
    x = multiply([x, se])
    return x

def senet_conv_block(input_tensor, kernel_size, filters,
                     stage, block, bias=False, strides=(2, 2)):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_eps = 0.0001

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv1_proj_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_proj"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias, strides=strides,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn",epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn",epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn" ,epsilon=bn_eps)(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    shortcut = Conv2D(filters3, (1, 1), use_bias=bias, strides=strides,
                      name=conv1_proj_name)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis,
                                  name=conv1_proj_name + "/bn",epsilon=bn_eps)(shortcut)

    m = layers.add([se, shortcut])
    m = Activation('relu')(m)
    return m

def senet_identity_block(input_tensor, kernel_size,
                         filters, stage, block, bias=False):
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1

    bn_eps = 0.0001

    conv1_reduce_name = 'conv' + str(stage) + "_" + str(block) + "_1x1_reduce"
    conv1_increase_name = 'conv' + str(stage) + "_" + str(
        block) + "_1x1_increase"
    conv3_name = 'conv' + str(stage) + "_" + str(block) + "_3x3"

    x = Conv2D(filters1, (1, 1), use_bias=bias,
               name=conv1_reduce_name)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=conv1_reduce_name + "/bn",epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same', use_bias=bias,
               name=conv3_name)(x)
    x = BatchNormalization(axis=bn_axis, name=conv3_name + "/bn",epsilon=bn_eps)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv1_increase_name, use_bias=bias)(x)
    x = BatchNormalization(axis=bn_axis, name=conv1_increase_name + "/bn",epsilon=bn_eps)(x)

    se = senet_se_block(x, stage=stage, block=block, bias=True)

    m = layers.add([se, input_tensor])
    m = Activation('relu')(m)

    return m

def SENET50(include_top=True, weights='vggface',
            input_tensor=None, input_shape=None,
            pooling=None,
            classes=8631):
    print("Using modified model")

    # input_shape = (224,224,3)
    input_shape = (None, None, 3) # Dynamic input tensor shape
    input = Input(shape=input_shape, batch_size=1)
    # input = Input(shape=input_shape, batch_size=1, dtype=tf.uint8)
    bn_axis = 3

    x = ChannelReversal()(input)
    x = Resizing(224, 224, interpolation='bilinear', name="Resize")(x)
    x = DepthwiseNormalization([91.4953, 103.8827, 131.0912])(x)

    x = Conv2D(
        64, (7, 7), use_bias=False, strides=(2, 2), padding='same',
        name='conv1/7x7_s2')(x)
    
    bn_eps = 0.0001
    x = BatchNormalization(axis=bn_axis, name='conv1/7x7_s2/bn',epsilon=bn_eps)(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = senet_conv_block(x, 3, [64, 64, 256], stage=2, block=1, strides=(1, 1))
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=2)
    x = senet_identity_block(x, 3, [64, 64, 256], stage=2, block=3)

    x = senet_conv_block(x, 3, [128, 128, 512], stage=3, block=1)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=2)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=3)
    x = senet_identity_block(x, 3, [128, 128, 512], stage=3, block=4)

    x = senet_conv_block(x, 3, [256, 256, 1024], stage=4, block=1)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=2)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=3)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=4)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=5)
    x = senet_identity_block(x, 3, [256, 256, 1024], stage=4, block=6)

    x = senet_conv_block(x, 3, [512, 512, 2048], stage=5, block=1)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=2)
    x = senet_identity_block(x, 3, [512, 512, 2048], stage=5, block=3)

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        output = Dense(classes, activation='softmax', name='classifier')(x)
    else:
        if pooling == 'avg':
            output = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            output = GlobalMaxPooling2D()(x)

    model = Model(input, output, name='vggface_senet50')

    # load weights
    if weights == 'vggface':
        if include_top:
            weights_path = get_file('rcmalli_vggface_tf_senet50.h5',
                                    utils.SENET50_WEIGHTS_PATH,
                                    cache_subdir=utils.VGGFACE_DIR)
        else:
            weights_path = get_file('rcmalli_vggface_tf_notop_senet50.h5',
                                    utils.SENET50_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir=utils.VGGFACE_DIR)
        model.load_weights(weights_path)

    return model
