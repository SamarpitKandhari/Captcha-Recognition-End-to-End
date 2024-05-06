import sys
import os
import numpy as np
import string
import matplotlib.pylab as plt
from PIL import Image
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from src.exception import CustomException
from src.logger import logging

class PredictPipeline:
    def __init__(self,filename):
        self.filename=filename

    def predict(self):
        try:
            characters= string.ascii_lowercase + "0123456789"

            model_path=os.path.join("model", "model.h5")
            model = load_model(model_path)
            logging.info("Model loaded")
            
            imagename = self.filename
            
            test_image = Image.open(imagename)
            test_image = np.array(test_image)
            test_image = np.reshape(test_image, (50, 200, 3))
            test_image = test_image / 255.0


            # test_image = image.load_img(imagename, target_size = (200,50),color_mode='rgb')
            # test_image = np.array(test_image)
            # test_image = np.reshape(test_image, (50, 200, 3))
            # test_image = test_image/255

            logging.info("Image Pre-Processed")

            test_image = np.expand_dims(test_image, axis = 0)
            pred=model.predict(test_image)
            logging.info("Prediction Done")
            
            result=[]
            print(pred)
            for i in range(5):
                n=np.argmax(pred[0][i])
                result.append(characters[n])
            
            print(result)

            return [{ "image" : result}]
            
        except Exception as e:
            raise CustomException(e,sys)
