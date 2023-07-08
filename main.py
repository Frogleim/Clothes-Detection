import os
import time
from core import age_gender_detection
from fastapi import FastAPI, File, UploadFile
from colorthief import ColorThief
from keras.models import load_model
from pydantic import BaseModel
from core import human_detection, fashion_mnist_predict
import cv2 as cv
import numpy as np
from pathlib import Path
import requests
from PIL import Image
import urllib.request
import io
import firebase_admin
from firebase_admin import credentials, storage
import glob
import urllib
import base64
from core.run import DeepFashion
from core.upload_images import upload_images
from core.db_data import data_storing
from core.run_models import predict_model, with_person


class Item(BaseModel):
    url: str
    user_id: str


class FinalItem(BaseModel):
    id: int


upload_url: str
img_urls: str
app = FastAPI()
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def clean_folder():
    file_path = './core/images/request_images/ready image/*'
    for f in glob.glob(file_path):
        os.remove(f)
    print('Cleaned successfully')


def download_image(image_url):
    urllib.request.urlretrieve(image_url, "./core/images/request_images/ready image/image.png")
    print('Image download successfully....')


def get_images(image_url):
    URL = urllib.request.urlretrieve(image_url)
    img = Image.open(URL[0])
    bytesIO = io.BytesIO()
    img.save(bytesIO, format='PNG')
    byteArr = bytesIO.getvalue()
    return byteArr


def colors_domination(file):
    color_thief = ColorThief(file)
    dominant_color = color_thief.get_color(quality=6)
    pallet_color = color_thief.get_palette(color_count=6)
    red, green, blue = None, None, None
    for colors in pallet_color:
        red, green, blue = colors
    if red > green and red > blue:
        return {'Colors': dominant_color, 'Dominant color': 'Dominant color is RED'}
    elif green > red and green > blue:
        return {'colors': dominant_color, 'dominant_color': 'Dominant color is GREEN'}
    else:
        return {'colors': dominant_color, 'dominant_color': 'Dominant color is BLUE'}


@app.post("/api/predict/")
async def get_image(item: Item):
    clean_folder()  # Cleaning everything
    image = get_images(item.url)
    person_detection = human_detection.PersonDetection(image=image)
    detect = person_detection.run()
    print(detect)
    dom_color = None
    if not detect:
        data = predict_model(image, item.user_id)
        folder_path = "./core/images/request_images/first step/"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                dom_color = colors_domination(file_path)
        return [{'result': data, 'person detection': 'false', 'dominant_color': dom_color}]
    else:
        data_person = with_person(image)
        gender = age_gender_detection.GenderDetection()
        gender_res = gender.age_gender_detector(item.url)

        folder_path = "./core/images/request_images/first step/"
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            if os.path.isfile(file_path):
                dom_color = colors_domination(file_path)
        return [{'result': data_person, 'person detection': 'true', 'dominant_color': dom_color, "gender": gender_res}]


@app.get('/')
async def create_item():
    return 'Welcome to Clothes Detection API'
