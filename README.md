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

- Finetuning 

~~~python
from keras.engine import Input, Model
from keras.layers import Flatten, Dense
from vggface import VGGFace

#custom parameters
nb_class = 2
hidden_dim = 512

#for theano backend image_input = Input(shape=(3,224, 224))
image_input = Input(shape=(224, 224, 3))
vgg_model = VGGFace(input_tensor=image_input, include_top=False)
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(image_input, out)
~~~

- Prediction: Tensorflow backend with 'tf' dimension ordering

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

- Prediction:  Theano backend with 'th' dimension ordering

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

The code that provided in this project is under MIT License.

If you find this project useful, please include reference link in your work.