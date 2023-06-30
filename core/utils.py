from yolov3_tf2.models import YoloV3
import numpy as np
import tensorflow as tf
import time
import cv2


def Draw_Bounding_Box(img, list_obj):
    try:
        img = img.numpy()  # convert tensor to numpy array
    except Exception:
        pass

    img = np.squeeze(img)

    img_width = img.shape[1]
    img_height = img.shape[0]

    color_yellow = [244 / 255, 241 / 255, 66 / 255]
    color_green = [66 / 255, 241 / 255, 66 / 255]
    color_red = [241 / 255, 66 / 255, 66 / 255]

    for obj in list_obj:
        x1 = int(round(obj['x1'] * img_width))
        y1 = int(round(obj['y1'] * img_height))
        x2 = int(round(obj['x2'] * img_width))
        y2 = int(round(obj['y2'] * img_height))

        if obj['label'] == 'short_sleeve_top':
            color = color_yellow
        elif obj['label'] == 'trousers':
            color = color_red
        else:
            color = color_green

        text = '{}: {:.2f}'.format(obj['label'], obj['confidence'])

        img = cv2.rectangle(img, (x1, y1), (x2, y2), color, 4)
        img = cv2.putText(img, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)

    return img


def Read_Img_2_Tensor(img_path):
    img = tf.image.decode_image(img_path, channels=3, dtype=tf.dtypes.float32)
    img = tf.expand_dims(img, 0)

    return img


def Load_Model():
    t1 = time.time()
    model = YoloV3(classes=13)
    model.load_weights(
        r'C:\Users\GSD Beast N10\Desktop\Projects\ml_models\deepfashion2_yolov3')  # TODO add DeepFashion model PATH
    t2 = time.time()
    print('Load DeepFashion2 Yolo-v3 from disk: {:.2f} sec'.format(t2 - t1))

    return model


def Save_Image(image_array, save_path):
    if image_array.dtype == 'float32':
        cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR) * 255)
    elif image_array.dtype == 'uint8':
        cv2.imwrite(save_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR))
    else:
        raise ValueError('Unrecognize type of image array: {}', image_array.dtype)

