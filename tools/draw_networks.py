from keras.utils import plot_model
from keras_vggface import VGGFace

model = VGGFace(model='vgg16')
plot_model(model, to_file='vgg16.png', show_shapes=True)

model = VGGFace(model='resnet50')
plot_model(model, to_file='resnet50.png', show_shapes=True)

model = VGGFace(model='senet50')
plot_model(model, to_file='senet50.png' ,show_shapes=True)