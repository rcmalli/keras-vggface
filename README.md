# keras-vggface

Oxford VGGFace  Implementation using Keras Functional Framework v2+

- This model is converted from original caffe network. (580 MB)
- It supports both Tensorflow and Theano backeds.
- You can also load only feature extraction layers with VGGFace(include_top=False) initiation (59MB).
- When you use it for the first time , weights are downloaded and stored in ~/.keras folder.

~~~bash

pip install keras_vggface

~~~

### News

- Project is now up-to-date with the new Keras version (2.0).

- Old Implementation is still available at 'keras1' branch.

### Library Versions

- Keras v2.0+
- Tensorflow 1.0+

### Example Usage


- Feature Extraction

~~~python
from keras.engine import  Model
from keras.layers import Input
from keras_vggface.vggface import VGGFace

# Convolution Features
vgg_model_conv = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

# FC7 Features
vgg_model = VGGFace() # pooling: None, avg or max
out = vgg_model.get_layer('fc7').output
vgg_model_fc7 = Model(vgg_model.input, out)

# After this point you can use your models as usual for both.
# ...

~~~



- Finetuning 

~~~python
from keras.engine import  Model
from keras.layers import Flatten, Dense, Input
from keras_vggface.vggface import VGGFace

#custom parameters
nb_class = 2
hidden_dim = 512

vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
last_layer = vgg_model.get_layer('pool5').output
x = Flatten(name='flatten')(last_layer)
x = Dense(hidden_dim, activation='relu', name='fc6')(x)
x = Dense(hidden_dim, activation='relu', name='fc7')(x)
out = Dense(nb_class, activation='softmax', name='fc8')(x)
custom_vgg_model = Model(vgg_model.input, out)

# Train your model as usual.
# ...
~~~

- Prediction

~~~python
import numpy as np
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace

# tensorflow
model = VGGFace()

# Change the image path with yours.
img = image.load_img('../image/ak.jpg', target_size=(224, 224))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
# TF order aka 'channel-last'
x = x[:, :, :, ::-1]
# TH order aka 'channel-first'
# x = x[:, ::-1, :, :]
# Zero-center by mean pixel
x[:, 0, :, :] -= 93.5940
x[:, 1, :, :] -= 104.7624
x[:, 2, :, :] -= 129.1863

preds = model.predict(x)
print('Predicted:', np.argmax(preds[0]))

~~~


### References

- [Keras Framework](www.keras.io)

- [Oxford VGGFace Website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

- [Related Paper](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)


### Licence 

Original models can be used for non-commercial research purposes under Creative Commons Attribution License.

The code that provided in this project is under MIT License.

If you find this project useful, please include reference link in your work.
