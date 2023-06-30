from .db_data import data_storing
from .human_detection import PersonDetection
from .fashion_mnist_predict import predict
import base64
import urllib.request
from PIL import Image
import io
import numpy as np
import cv2 as cv
from core.run import DeepFashion
from .utils import Read_Img_2_Tensor, Load_Model, Save_Image
from .app import Detect_Clothes_and_Crop
import glob
import os


model = Load_Model()


def get_images(image_url):
    URL = urllib.request.urlretrieve(image_url)
    img = Image.open(URL[0])
    bytesIO = io.BytesIO()
    img.save(bytesIO, format='PNG')
    byteArr = bytesIO.getvalue()
    return byteArr


def convert_img(img):
    img_np = np.frombuffer(img, np.uint8)
    image = cv.imdecode(img_np, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    return image


def convert_img_ndarray(img):
    image = cv.imdecode(img, cv.IMREAD_UNCHANGED)
    image = cv.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    return image


def clean_folder():
    file_path = './core/images/request_images/first step/*'
    for f in glob.glob(file_path):
        os.remove(f)
    print('Cleaned successfully')


def predict_model(img, users_id, count=None):
    images = convert_img(img)
    print(predict(images))
    file_name = f'crop_0.png'
    print('Saving image...')
    Save_Image(images, f'./core/images/request_images/first step/{file_name}')
    return predict(images)


def with_person(img):
    img_tensor = Read_Img_2_Tensor(img)
    img_crop = Detect_Clothes_and_Crop(img_tensor, model)
    keys = [item.keys() for item in img_crop]
    keys = [key for key_list in keys for key in key_list]

    val = [val_item.values() for val_item in img_crop]
    val = [key for val_list in val for key in val_list]
    i = 0
    clean_folder()
    for images in val:
        i += 1
        file_name = f'crop_{i}.png'
        print('Saving image...')
        Save_Image(images, f'./core/images/request_images/first step/{file_name}')

    return keys


