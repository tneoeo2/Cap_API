import sys
import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from utils.img_processing import encode_single_sample
from utils.img_processing import decode_batch_predictions
from utils.img_module import ImageModule as im

# BW_MODEL_PATH = 'models/bw_captcha_model/'

class LoadModel():
    '''
        모델 하나당 Load_Model 하나씩 생성할 것
    '''
    
    def __init__(self, model_path):
        self.model_path = model_path
        
    def load_bw_model(self, summary=True):
        # Load the saved model
        print("Loading model: {}".format(self.model_path))
        loaded_model = keras.models.load_model(self.model_path)
        loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
        if summary is True:
            print(loaded_model.summary())
        return loaded_model
        

# tf_model = LoadModel(BW_MODEL_PATH).load_bw_model()
# print("Model loading...")
# print(tf_model.summary())

