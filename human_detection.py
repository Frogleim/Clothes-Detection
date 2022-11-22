import cv2
import imutils
from io import BytesIO
import numpy as np
from keras.models import load_model
from run import DeepFashion
from firebase_admin import credentials, initialize_app, storage
import json


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
        new_model = load_model('./model')
        if detection is None:
            print('Person not detected')
            predict_image = read_imagefile(image)
            prediction = new_model.predict(predict_image)
            predicted_label = self.class_names[np.argmax(prediction[0])]
            return {'person_detection': 'false', 'Objects in image': f'{predicted_label}'}

        else:

            deep_fashion_model = DeepFashion()
            deep_fashion_model.final_predict(self.image)
            file = open(r'./objects.json')
            data = json.load(file)
            print('Person was detected!')

            # return {'Person detection': 'True', 'Objects in image': f'{data}'}

            return data


if __name__ == '__main__':
    image_path = r'C:\Users\GSD Beast N10\Desktop\Projects\wallet\images\dress.jpg'
    person_detection = PersonDetection(image_path=image_path)
    person_detection.run()
