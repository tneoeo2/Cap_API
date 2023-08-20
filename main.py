import io
import threading
import uvicorn
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

from tempfile import NamedTemporaryFile
from typing import IO
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from utils.img_module import ImageModule
from utils.img_processing import encode_single_sample
from utils.img_processing import decode_batch_predictions
from utils.load_model import LoadModel
app = FastAPI()
BW_MODEL_PATH = 'models/bw_captcha_model/'
model_list = {}
im = ImageModule()

# async def save_file(file: IO):   #업로드 이미지 저장코드
#     with NamedTemporaryFile("wb", delete=False) as tempfile:
#         tempfile.write(file.read())
#         return tempfile.name

# PIL 이미지를 TensorFlow의 tf.Tensor 형식으로 변환하는 함수
def pil_to_tensor(pil_image):
    # PIL 이미지를 이진 데이터로 변환
    with io.BytesIO() as byte_io:
        pil_image.save(byte_io, format='PNG')
        image_data = byte_io.getvalue()

    # 이진 데이터를 tf.Tensor 형식으로 변환
    image_tensor = tf.io.decode_png(image_data, channels=3)  # channels 값은 이미지의 채널 수에 맞게 설정

    return image_tensor


@app.post("/read_captcha")
async def read_captcha(file: UploadFile = File(...)):
    bw_cap_model = model_list['bw_cap_model'] 
    # path = await save_file(file.file)
    contents = await file.read()  #Bytes형태로 이미지 반환
    #전처리 수행
    image = Image.open(file.file)
    # print("image : ", image, "\n", type(image))
    #PIL 이미지를 TensorFlow의 tf.Tensor 형식으로 변환
    image_tensor = pil_to_tensor(image)
    # print("image_tensor : ", image_tensor, "\n", type(image_tensor))
   
    test_img = encode_single_sample(image_tensor.numpy(), img_height=50, img_width=200)
    test_img_array = np.array([test_img['image'].numpy()] * 1)
    # 모델테스트 
    preds = bw_cap_model.predict(test_img_array)
    pred_texts = decode_batch_predictions(preds)

    # print("preds: ", preds)
    # print("preds: ", pred_texts)

    return {"filename": file.filename, "preds": pred_texts}

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8888)   #이 밑의 코드는 실행되지 않는다.
    
if __name__ == "__main__":
    server_thread = threading.Thread(target=start_server)
    # 서버가 실행되는 동안 다른 코드 실행 가능
    server_thread.start()
    print("Server started.")
    
    tf_model = LoadModel(BW_MODEL_PATH).load_bw_model(summary=True)
    model_list['bw_cap_model'] = tf_model
    print("Model List: {}".format(model_list))
    
    # 서버 스레드가 종료되길 기다림
    server_thread.join()
    print("Server stopped.")