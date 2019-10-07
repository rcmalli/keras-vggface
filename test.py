import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest


class VGGFaceTests(unittest.TestCase):

    def testVGG16(self):
        keras.backend.image_data_format()
        model = VGGFace(model='vgg16')
        img = image.load_img('image/ajb.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=1)
        preds = model.predict(x)
        # print ('\n', "VGG16")
        # print('\n',preds)
        # print('\n','Predicted:', utils.decode_predictions(preds))
        self.assertIn('A.J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.9790116, places=3)

    def testRESNET50(self):
        keras.backend.image_data_format()
        model = VGGFace(model='resnet50')
        img = image.load_img('image/ajb.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = model.predict(x)
        # print ('\n',"RESNET50")
        # print('\n',preds)
        # print('\n','Predicted:', utils.decode_predictions(preds))
        self.assertIn('A._J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.91819614, places=3)

    def testSENET50(self):
        keras.backend.image_data_format()
        model = VGGFace(model='senet50')
        img = image.load_img('image/ajb.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x, version=2)
        preds = model.predict(x)
        # print ('\n', "SENET50")
        # print('\n',preds)
        # print('\n','Predicted:', utils.decode_predictions(preds))
        self.assertIn('A._J._Buckley', utils.decode_predictions(preds)[0][0][0])
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.9993529, places=3)


if __name__ == '__main__':
    unittest.main()
