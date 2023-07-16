import os
import cv2
import tensorflow as tf
import numpy as np

from glob import glob
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model
from utils.img_module import ImageModule


#캡차코드에 사용되는 알파벳 
characters =  ['a', 'b', 'c', 'd', 'e', 'k', 'l', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'w', 'x', 'z']
max_length = 6  #캡차 글자수
im = ImageModule()

# 문자를 숫자로 바꿉니다.
char_to_num = layers.experimental.preprocessing.StringLookup(
    vocabulary=list(characters), num_oov_indices=0, mask_token=None
)

# 숫자를 문자로 바꿉니다.
num_to_char = layers.experimental.preprocessing.StringLookup(
    vocabulary=char_to_num.get_vocabulary(),num_oov_indices=0, mask_token=None, invert=True
)


def encode_single_sample(img_path, label=None, img_height=50, img_width=200):
    '''
    img : 이미지(Bytes)
    label : 이미지 라벨(라벨링된 이미지일 경우)
    img_height : 이미지 세로 사이즈
    img_width : 이미지 가로 사이즈
    '''
    # 1. 이미지를 불러옵시다.
    img = tf.io.read_file(img_path)
    img = im.reverse_bg(img)
    # 2. png 이미지로 변환하고, 해당 이미지를 grayscale로 변환합시다.
    _, edcoded_img = cv2.imencode('.png', img)
    byte_data = edcoded_img.tobytes()
    img = tf.convert_to_tensor(byte_data)   #nparray 데이터 바이트타입으로 변환
    img = tf.io.decode_png(img, channels=1)
    
    # 3. [0, 255]의 정수 범위를 [0, 1]의 실수 범위로 바꿉시다.
    img = tf.image.convert_image_dtype(img, tf.float32)
    
    # 4. 위에서 정한 이미지 사이즈로 resize합시다.
    img = tf.image.resize(img, [img_height, img_width])
    
    # 5. 이미지와 가로와 세로를 뒤바꿉시다.
    # 우리는 이미지의 가로와 시간차원을 대응시키고 싶기 때문입니다.
    img = tf.transpose(img, perm=[1, 0, 2])
    
    # 6. 라벨값의 문자를 숫자로 바꿉시다.
    if label is not None:
        label = char_to_num(tf.strings.unicode_split(label, input_encoding="UTF-8"))
    
    # 7. 우리의 모델은 두개의 input을 받기 때문에
    # dictionary에 담아서 return 합니다.
    return {"image": img, "label": label}


# 데이터 디코딩(예측결과 디코딩할때 사용 -> 문자로 바꾼다)
def decode_batch_predictions(pred):
    input_len = np.ones(pred.shape[0]) * pred.shape[1]
    # Use greedy search. For complex tasks, you can use beam search
    results = keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0][
        :, :max_length
    ]
    # Iterate over the results and get back the text
    output_text = []
    for res in results:
        res = tf.strings.reduce_join(num_to_char(res)).numpy().decode('utf-8')
        output_text.append(res)
    return output_text