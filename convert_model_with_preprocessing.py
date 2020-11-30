import numpy as np
from keras_vggface import VGGFace, utils
from tensorflow.keras.preprocessing import image
import unittest
import tensorflow as tf

# # Using the old way
# model = VGGFace(model='senet50')
# img = image.load_img('image/ajb-resized.jpg', target_size=(224,224), interpolation="bilinear")
# x = image.img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = utils.preprocess_input(x, version=2)
# preds = model.predict(x)
# print('Predicted:', utils.decode_predictions(preds))
# # Output of normal:               [[["b' A._J._Buckley'", 0.91385096], ["b' Guy_Garvey'", 0.009176245], ["b' Jeff_Corwin'", 0.008781389], ["b' Michael_Voltaggio'", 0.0073467665], ["b' Nick_Frost'", 0.0065856054]]]
# # Output of custom preprocessing: [[["b' A._J._Buckley'", 0.91558367], ["b' Guy_Garvey'", 0.009039231], ["b' Jeff_Corwin'", 0.008346532], ["b' Michael_Voltaggio'", 0.0071733994], ["b' Nick_Frost'", 0.006603726]]]

# # Using custom preprocessing layers
# model = VGGFace(model='senet50')
# img = image.load_img('image/ajb-resized.jpg')
# x = image.img_to_array(img, dtype="uint8")
# x = np.expand_dims(x, axis=0)
# preds = model.predict(x)
# print('\n', preds)
# print('\n','Predicted:', utils.decode_predictions(preds))

images_directory_path = "/Users/ben/butter/repos/popsa/datasets/2020-06-12/images/graeme/"
filename = "00116D7D-A054-4A56-8296-E3535FBA7610_L0_001_0.png"
model = VGGFace(model="senet50", pooling="avg", include_top=False)
img = image.load_img(images_directory_path + filename)
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
embeddings = model.predict(x)
print("Embeddings: ", embeddings)


# # Conversion
# model = VGGFace(model="senet50", pooling="avg", include_top=False)
# converter = tf.lite.TFLiteConverter.from_keras_model(model)
# converter.optimizations = []
# # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# # Set the input and output tensors to uint8 (APIs added in r2.3)
# # converter.inference_input_type = tf.uint8

# converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
#                                        tf.lite.OpsSet.SELECT_TF_OPS]

# tflite_model = converter.convert()
# model_filename = 'VggFace2SeNet.tflite'
# with open(model_filename, 'wb') as f:
#   f.write(tflite_model)