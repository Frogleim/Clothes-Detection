import tensorflow as tf
from io import BytesIO
import cv2
import numpy as np
from keras.models import load_model

model_path = r'C:\Users\OMEN\Desktop\mobile app\ml_models'
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']


def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (28, 28))
    image = image.astype('float32')
    image = image.reshape(1, 28, 28, 1)
    image = 255 - image
    image /= 255
    return image


def predict(image):
    new_model = load_model(model_path + '\model')
    prediction = new_model.predict(image)
    predicted_label = class_names[np.argmax(prediction[0])]
    return predicted_label


if __name__ == '__main__':
    image_path = 'C:\\Users\\GSD Beast N10\\Desktop\\Projects\\lookAt\\request_images\\ready image\\crop_image_1.jpg'
    res = predict(image_path)
    print(res)