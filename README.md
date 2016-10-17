# keras-vggface

Oxford VGGFace  Implementation using Keras Functional Framework 1.1

- This model is converted from original caffe network. (580 MB)
- It supports both Tensorflow and Theano backeds.
- You can also load only feature extraction layers with VGGFace(include_top=False) initiation (59MB).
- When you use it for the first time , weights are downloaded and stored in ~/.keras folder.


### Library Versions

- Keras v1.1
- Tensorflow v10
- Theano v0.8.2

### Example Usage

- Tensorflow backend with 'tf' dimension ordering

~~~python
from scipy import misc
import copy
import numpy as np
from vggface import VGGFace

model = VGGFace()

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

~~~

- Theano backend with 'th' dimension ordering

~~~python
from scipy import misc
import copy
import numpy as np
from vggface import VGGFace

model = VGGFace()

im = misc.imread('../image/ak.jpg')
im = misc.imresize(im, (224, 224)).astype(np.float32)
aux = copy.copy(im)
im[:, :, 0] = aux[:, :, 2]
im[:, :, 2] = aux[:, :, 0]
# Remove image mean
im[:, :, 0] -= 93.5940
im[:, :, 1] -= 104.7624
im[:, :, 2] -= 129.1863
im = np.transpose(im, (2, 0, 1))
im = np.expand_dims(im, axis=0)

res = model.predict(im)
print np.argmax(res[0])
~~~


### References

- [Keras Framework](www.keras.io)

- [Oxford VGGFace Website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

- [Related Paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)


###Licence 

Original models can be used for non-commercial research purposes under Creative Commons Attribution License.

The Code that provided in this project is under MIT License.