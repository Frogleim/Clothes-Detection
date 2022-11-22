import os
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"


class RemoveBG:

    def __init__(self):
        self.H = 512
        self.W = 512

    @staticmethod
    def create_dir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def read_model():

        with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
            model = tf.keras.models.load_model(r"./model.h5")
            return model

    def remove_bg(self, path):
        self.create_dir("remove_bg")

        data_x = glob(path)
        for path in tqdm(data_x, total=len(data_x)):
            """ Extracting name """
            name = path.split("/")[-1].split(".")[0]

            """ Read the image """
            image = cv2.imread(path, cv2.IMREAD_COLOR)
            h, w, _ = image.shape
            x = cv2.resize(image, (self.W, self.H))
            x = x / 255.0
            x = x.astype(np.float32)
            x = np.expand_dims(x, axis=0)
            model = self.read_model()
            y = model.predict(x)[0]
            y = cv2.resize(y, (w, h))
            y = np.expand_dims(y, axis=-1)
            y = y > 0.5

            photo_mask = y
            background_mask = np.abs(1 - y)
            masked_photo = image * photo_mask
            background_mask = np.concatenate([background_mask, background_mask, background_mask], axis=-1)
            background_mask = background_mask * [0, 0, 255]
            final_photo = masked_photo + background_mask
            cv2.imwrite(f"remove_bg/{name}.png", final_photo)



