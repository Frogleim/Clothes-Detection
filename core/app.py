import time

import cv2
import numpy as np
import tensorflow as tf

from .utils import Read_Img_2_Tensor, Load_Model, Draw_Bounding_Box

img_crop = None


def Detect_Clothes(image, model_yolov3, eager_execution=True):
    """Detect clothes in an image using Yolo-v3 model trained on DeepFashion2 dataset"""
    img = tf.image.resize(image, (416, 416))

    t1 = time.time()
    if eager_execution == True:
        boxes, scores, classes, nums = model_yolov3(img)
        # change eager tensor to numpy array
        boxes, scores, classes, nums = boxes.numpy(), scores.numpy(), classes.numpy(), nums.numpy()
    else:
        boxes, scores, classes, nums = model_yolov3.predict(img)
    t2 = time.time()
    print('Yolo-v3 feed forward: {:.2f} sec'.format(t2 - t1))

    class_names = ['T-shirt', 'Pullover(long sleeve top)', 'short_sleeve_outwear', 'long_sleeve_outwear',
                   'vest', 'sling', 'shorts', 'trousers', 'skirt', 'Short Sleeve Dress',
                   'Long Sleeve Dress', 'vest_dress', 'sling_dress']

    list_obj = []
    for i in range(nums[0]):
        obj = {'label': class_names[int(classes[0][i])], 'confidence': scores[0][i]}
        obj['x1'] = boxes[0][i][0]
        obj['y1'] = boxes[0][i][1]
        obj['x2'] = boxes[0][i][2]
        obj['y2'] = boxes[0][i][3]
        list_obj.append(obj)

    return list_obj


def Detect_Clothes_and_Crop(img_tensor, model, threshold=0.5):
    global img_crop

    list_object = Detect_Clothes(img_tensor, model)
    img_crop_list = []
    img = np.squeeze(img_tensor.numpy())
    img_width = img.shape[1]
    img_height = img.shape[0]
    for obj in list_object:

        if obj['label'] and obj['confidence'] > threshold:
            img_crop = img[int(obj['y1'] * img_height):int(obj['y2'] * img_height),
                       int(obj['x1'] * img_width):int(obj['x2'] * img_width), :]
            img_array = {obj['label']: img_crop}

            img_crop_list.append(img_array)

    return img_crop_list




