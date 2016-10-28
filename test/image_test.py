from vggface import VGGFace
from scipy import misc
import copy
import numpy as np

if __name__ == '__main__':

    model = VGGFace(weights=None)
    model.load_weights('../temp/weight/rcmalli_vggface_tf_weights_tf_ordering.h5')
    print 'model loaded.'
    im = misc.imread('../image/ak.jpg')
    im = misc.imresize(im, (224, 224)).astype(np.float32)
    aux = copy.copy(im)
    im[:, :, 0] = aux[:, :, 2]
    im[:, :, 2] = aux[:, :, 0]
    # Remove image mean
    im[:, :, 0] -= 93.5940
    im[:, :, 1] -= 104.7624
    im[:, :, 2] -= 129.1863
    im = np.expand_dims(im, axis=0)

    res = model.predict(im)
    print np.argmax(res[0])