# keras-vggface [![Build Status](https://travis-ci.org/rcmalli/keras-vggface.svg?branch=master)](https://travis-ci.org/rcmalli/keras-vggface) [![PyPI Status](https://badge.fury.io/py/keras-vggface.svg)](https://badge.fury.io/py/keras-vggface) [![PyPI Status](https://pepy.tech/badge/keras-vggface)](https://pepy.tech/project/keras-vggface)

Oxford VGGFace  Implementation using Keras Functional Framework v2+

- Models are converted from original caffe networks.
- It supports only Tensorflow backend.
- You can also load only feature extraction layers with VGGFace(include_top=False) initiation.
- When you use it for the first time , weights are downloaded and stored in ~/.keras/models/vggface folder.
- If you don't know where to start check the [blog posts](https://github.com/rcmalli/keras-vggface#projects--blog-posts) that are using this library.

~~~bash
# Most Recent One (Suggested)
pip install git+https://github.com/rcmalli/keras-vggface.git
# Release Version
pip install keras_vggface
~~~


### Library Versions

- Keras v2.2.4
- Tensorflow v1.14.0
- **Warning: Theano backend is not supported/tested for now.**

### Example Usage

#### Available Models

```python

from keras_vggface.vggface import VGGFace

# Based on VGG16 architecture -> old paper(2015)
vggface = VGGFace(model='vgg16') # or VGGFace() as default

# Based on RESNET50 architecture -> new paper(2017)
vggface = VGGFace(model='resnet50')

# Based on SENET50 architecture -> new paper(2017)
vggface = VGGFace(model='senet50')

```


#### Feature Extraction
 
- Convolution Features

    ```python
    from keras.engine import  Model
    from keras.layers import Input
    from keras_vggface.vggface import VGGFace

    # Convolution Features
    vgg_features = VGGFace(include_top=False, input_shape=(224, 224, 3), pooling='avg') # pooling: None, avg or max

    # After this point you can use your model to predict.
    # ...

    ```


- Specific Layer Features

    ```python
    from keras.engine import  Model
    from keras.layers import Input
    from keras_vggface.vggface import VGGFace

    # Layer Features
    layer_name = 'layer_name' # edit this line
    vgg_model = VGGFace() # pooling: None, avg or max
    out = vgg_model.get_layer(layer_name).output
    vgg_model_new = Model(vgg_model.input, out)

    # After this point you can use your model to predict.
    # ...

    ```



#### Finetuning

- VGG16

    ```python
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
    ```

- RESNET50 or SENET50

    ```python
    from keras.engine import  Model
    from keras.layers import Flatten, Dense, Input
    from keras_vggface.vggface import VGGFace

    #custom parameters
    nb_class = 2

    vgg_model = VGGFace(include_top=False, input_shape=(224, 224, 3))
    last_layer = vgg_model.get_layer('avg_pool').output
    x = Flatten(name='flatten')(last_layer)
    out = Dense(nb_class, activation='softmax', name='classifier')(x)
    custom_vgg_model = Model(vgg_model.input, out)

    # Train your model as usual.
    # ...
    ```



#### Prediction

- Use `utils.preprocess_input(x, version=1)` for VGG16
- Use `utils.preprocess_input(x, version=2)` for RESNET50 or SENET50


    ```python
    import numpy as np
    from keras.preprocessing import image
    from keras_vggface.vggface import VGGFace
    from keras_vggface import utils

    # tensorflow
    model = VGGFace() # default : VGG16 , you can use model='resnet50' or 'senet50'

    # Change the image path with yours.
    img = image.load_img('../image/ajb.jpg', target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = utils.preprocess_input(x, version=1) # or version=2
    preds = model.predict(x)
    print('Predicted:', utils.decode_predictions(preds))
    ```


### References

- [Keras Framework](www.keras.io)

- [Oxford VGGFace Website](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/)

- [Related Paper 1](http://www.robots.ox.ac.uk/~vgg/publications/2015/Parkhi15/parkhi15.pdf)

- [Related Paper 2](http://www.robots.ox.ac.uk/~vgg/publications/2018/Cao18/cao18.pdf)

### Licence 

- Check Oxford Webpage for the license of the original models.

- The code that provided in this project is under MIT License.

### Projects / Blog Posts

If you find this project useful, please include reference link in your work. You can create PR's to this document with your project/blog link.

- [Live Face Identification with pre-trained VGGFace2 model](https://www.dlology.com/blog/live-face-identification-with-pre-trained-vggface2-model/)

- [How to Perform Face Recognition With VGGFace2 in Keras](https://machinelearningmastery.com/how-to-perform-face-recognition-with-vggface2-convolutional-neural-network-in-keras/)

- [An extremely small FaceRecog project for extreme beginners, and a few thoughts on the future](https://kevincodeidea.wordpress.com/2020/01/14/an-extremely-small-facerecog-project-for-extreme-beginners-and-a-few-thoughts-on-future-part-ii-transfer-learning-and-keras/)

