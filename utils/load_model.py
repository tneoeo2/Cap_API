import sys
import os
import cv2
import numpy as np

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

from utils.img_processing import encode_single_sample
from utils.img_processing import decode_batch_predictions
from utils.img_module import ImageModule as im

# BW_MODEL_PATH = 'models/bw_captcha_model/'

class LoadModel():
    '''
        모델 하나단 Load_Model 하나씩 생성할 것
    '''
    
    def __init__(self, model_path):
        self.model_path = model_path
        
  
        
    def load_bw_model(self, summary=True):
        # Load the saved model
        print("Loading model: {}".format(self.model_path))
        DEFAULT_FUNCTION_KEY = "serving_default"
        loaded_model = tf.saved_model.load(self.model_path)
        model = self.build_model(loaded_model)
        
        print(f'loaded_model : {model}')
        model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
        if summary is True:
            print(model.summary())
        return model
    
    
    def build_model(self, loaded):
        x = tf.keras.layers.Input(shape=(200, 50, 1), name='input_x')
        # 불러온 것을 케라스 레이어로 감쌉니다.
        keras_layer = hub.KerasLayer(loaded, trainable=True)(x)
        model = tf.keras.Model(x, keras_layer)
    
        return model

    # def load_bw_model(self, summary=True):
    #     # Load the saved model
    #     print("Loading model: {}".format(self.model_path))
    #     loaded_model = tf.saved_model.load(self.model_path)
    #     print(f'loaded_model : {loaded_model}')
    #     loaded_model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False))
    #     if summary is True:
    #         print(loaded_model.summary())
    #     return loaded_model
        

# tf_model = LoadModel(BW_MODEL_PATH).load_bw_model()
# print("Model loading...")
# print(tf_model.summary())

