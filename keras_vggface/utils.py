import numpy as np
from keras import backend as K
from keras.utils.data_utils import get_file

""" functions are mostly taken and modified from keras/applications (https://github.com/fchollet/keras/tree/master/keras/applications) """

WEIGHTS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_v2.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_tf_notop_v2.h5'
LABELS_PATH = 'https://github.com/rcmalli/keras-vggface/releases/download/v2.0/rcmalli_vggface_labels.npy'

LABELS = None


def preprocess_input(x, data_format=None):
    if data_format is None:
        data_format = K.image_data_format()
    assert data_format in {'channels_last', 'channels_first'}

    if data_format == 'channels_first':
        x = x[:, ::-1, ...]
        x[:, 0, :, :] -= 93.5940
        x[:, 1, :, :] -= 104.7624
        x[:, 2, :, :] -= 129.1863
    else:
        x = x[..., ::-1]
        x[..., 0] -= 93.5940
        x[..., 1] -= 104.7624
        x[..., 2] -= 129.1863
    return x


def decode_predictions(preds, top=5):
    global LABELS
    if len(preds.shape) != 2 or preds.shape[1] != 2622:
        raise ValueError('`decode_predictions` expects '
                         'a batch of predictions '
                         '(i.e. a 2D array of shape (samples, 2622)). '
                         'Found array with shape: ' + str(preds.shape))
    if LABELS is None:
        fpath = get_file('rcmalli_vggface_labels.json',
                         LABELS_PATH,
                         cache_subdir='models')
        LABELS = np.load(fpath)
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [[str(LABELS[i]), pred[i]] for i in top_indices]
        result.sort(key=lambda x: x[1], reverse=True)
        results.append(result)
    return results
