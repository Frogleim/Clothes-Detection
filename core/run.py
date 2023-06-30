from .app import Detect_Clothes_and_Crop
from .utils import Read_Img_2_Tensor, Load_Model, Save_Image
import json
import os
import tensorflow as tf
import glob


class DeepFashion:

    def __init__(self):
        self.model = Load_Model()
        self.images_data = []
        self.images_label_dict = {}

    def run(self, img):
        print('First Step begins....')
        files = glob.glob(r'./core/images/request_images/first step/*.png')
        for f in files:
            os.remove(f)
        img_tensor = Read_Img_2_Tensor(img)
        img_crop = Detect_Clothes_and_Crop(img_tensor, self.model)
        return img_crop
