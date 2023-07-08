import cv2
import imutils
from io import BytesIO
import numpy as np
from keras.models import load_model
from core.run import DeepFashion
import tensorflow as tf
from firebase_admin import credentials, initialize_app, storage
import json
import cv2 as cv
from .fashion_mnist_predict import predict


def detect(regions, images):
    for (x, y, w, h) in regions:
        detection = cv2.rectangle(images, (x, y),
                                  (x + w, y + h),
                                  (0, 0, 255), 2)
        return detection


class PersonDetection:

    def __init__(self, image):
        self.HOGCV = cv2.HOGDescriptor()
        self.HOGCV.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.image = image
        self.class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        self.model_path = r'C:\Users\OMEN\Desktop\mobile app\ml_models'

    def convert_image(self):

        image_stream = BytesIO(self.image)
        file_bytes = np.asarray(bytearray(image_stream.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)

        return image

    def read_image(self):

        image = self.convert_image()
        image = imutils.resize(image,
                               width=min(400, image.shape[1]))
        (regions, _) = self.HOGCV.detectMultiScale(image,
                                                   winStride=(4, 4),
                                                   padding=(4, 4),
                                                   scale=1.05)
        return regions, image

    def run(self):
        regions, image = self.read_image()
        detection = detect(regions=regions, images=image)
        if detection is None:
            print('Person not detected')
            return False

        else:
            return True


