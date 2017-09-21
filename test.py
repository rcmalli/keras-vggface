import numpy as np
from keras_vggface import VGGFace
from keras.preprocessing import image
from keras_vggface import utils
import keras
import unittest



class VGGTests(unittest.TestCase):

    def testModelInit(self):
        model = VGGFace()
        self.assertIsNotNone(model)
    def testTFwPrediction(self):
        keras.backend.set_image_dim_ordering('tf')
        model = VGGFace()
        img = image.load_img('image/ak.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x)
        preds = model.predict(x)
        print('Predicted:', utils.decode_predictions(preds))
        self.assertIn(utils.decode_predictions(preds)[0][0][0], 'Aamir_Khan')
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.94938219)
    def testTHPrediction(self):
        keras.backend.set_image_dim_ordering('th')
        model = VGGFace()
        img = image.load_img('image/ak.jpg', target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = utils.preprocess_input(x)
        preds = model.predict(x)
        print('Predicted:', utils.decode_predictions(preds))
        self.assertIn(utils.decode_predictions(preds)[0][0][0], 'Aamir_Khan')
        self.assertAlmostEqual(utils.decode_predictions(preds)[0][0][1], 0.94938219)

if __name__ == '__main__':
    unittest.main()



