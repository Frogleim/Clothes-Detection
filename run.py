from app import Detect_Clothes_and_Crop
from utils import Read_Img_2_Tensor, Load_Model, Save_Image
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
        files = glob.glob(r'C:\Users\GSD Beast N10\Desktop\Projects\lookAt\request_images\first step\*.jpg')
        for f in files:
            os.remove(f)
        img_tensor = Read_Img_2_Tensor(img)
        img_crop = Detect_Clothes_and_Crop(img_tensor, self.model)
        i = 0
        for images in img_crop:
            for labels, crop_images in images.items():
                i += 1
                file_name = f'crop_{i}.jpg'
                Save_Image(crop_images, f'C:\\Users\\GSD Beast N10\\Desktop\\Projects\\lookAt\\request_images\\first step'
                                        f'\\{file_name}')

    def final_predict(self, img):
        files = glob.glob(r'C:\Users\GSD Beast N10\Desktop\Projects\lookAt\request_images\ready image\*.jpg')
        for f in files:
            os.remove(f)
        self.run(img)
        i = 0
        crop_images_path = r'C:\Users\GSD Beast N10\Desktop\Projects\lookAt\request_images\first step'
        for images in os.listdir(crop_images_path):
            if images.endswith('jpg'):
                images_path = f'{crop_images_path}\{images}'
                img_raw = tf.io.read_file(images_path)

                img_tensor = Read_Img_2_Tensor(img_raw)
                print(img_tensor)
                img_crop = Detect_Clothes_and_Crop(img_tensor, self.model)
                print(type(img_crop))
                for crop_images in img_crop:
                    for labels, crop_img in crop_images.items():
                        i += 1
                        file_name = f'crop_ready_{i}.jpg'
                        self.images_label_dict = {"id": i, "img_path": f'./request_images/ready_image/'}
                        self.images_data.append(self.images_label_dict)
                        with open(r'./objects.json', 'w') as savefile:
                            json.dump(self.images_label_dict, savefile)
                        Save_Image(crop_img, f'./request_images/ready image/{file_name}')


if __name__ == '__main__':
    deep_model = DeepFashion()
    image = input('Enter image path: ')
    deep_model.run()
