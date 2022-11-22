import uvicorn
from fastapi import FastAPI
from colorthief import ColorThief
from keras.models import load_model
from pydantic import BaseModel
from human_detection import PersonDetection
import cv2 as cv
import numpy as np
from pathlib import Path
import requests
from remove_bg import RemoveBG
from PIL import Image
import urllib.request
import io
import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import glob


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


def read_image(image_path):
    image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
    image = cv.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    return image


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


def upload_file(images_path, users_ids, count):
    global upload_url, img_urls

    for i in range(0, count):

        print(i)
        if not firebase_admin._apps:
            serviceAccount = './serviceAccount.json'
            cred = credentials.Certificate(serviceAccount)
            firebase_admin.initialize_app(cred, {'storageBucket': 'lookat-5d200.appspot.com'})
            bucket = storage.bucket()
            blob = bucket.blob(f'users_images/{users_ids}/users_predicted_images_{i}')
            blob.upload_from_filename(images_path)
            upload_url = blob.public_url
            img_name = f'users_predicted_images_{i}'
            img_url = f'https://firebasestorage.googleapis.com/v0/b/lookat-5d200.appspot.com/o/users_images' \
                      f'%2FZESTFU29upcX97tK6FdBIEKrhH42%2F{img_name}'

            r = requests.get(img_url)
            img_urls = img_url + '?alt=media&token=' + r.json()['downloadTokens']
            return img_urls


@app.post('/items')
async def create_item(item: Item):
    return item.url


@app.post("/api/upload/")
async def get_image(item: Item):
    image = get_images(item.url)
    person_detection = PersonDetection(image=image)
    person_check = person_detection.run()
    user_id = item.user_id

    image_path = f'./request_images/ready image/*.jpg'
    count = 0
    img_urls = None
    for files in glob.glob(image_path):
        count += 1
        try:
            img_urls = upload_file(users_ids=user_id, images_path=files, count=count)
            print(img_urls)
            print('Successfully uploaded...')
        except Exception as e:
            print(f'Failed!\n{e}')
    return [{'message': 'done', 'upload_url': img_urls, 'founded_objects': person_check}]


@app.post('/api/predict/')
async def predict_image(id: FinalItem):
    images_path = Path(f'./request_images/ready image/crop_ready_{str(id.id)}.jpg')
    images_name = Path(f'./request_images/ready image/crop_ready_{str(id.id)}.jpg').stem
    if images_path.is_file():
        predict_image_path = f'./request_images/ready image/{images_name}.jpg'
        new_model = load_model('./model')
        remove_bg = RemoveBG()
        remove_bg.remove_bg(predict_image_path)
        new_path = f'remove_bg/{images_name}.png'
        image = read_image(new_path)
        color_domination = colors_domination(predict_image_path)
        prediction = new_model.predict(image)
        predicted_label = class_names[np.argmax(prediction[0])]
        return [{'result': predicted_label, 'image_url': predict_image_path, 'colors_domination': color_domination}]
    else:
        return {{'Result': 'Wrong image id'}}


if __name__ == '__main__':
    uvicorn.run(app, port=50162)
