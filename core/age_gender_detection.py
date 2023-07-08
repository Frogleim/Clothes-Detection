import urllib.request as urllib
import cv2
import numpy as np


def read_image_from_url(url):
    req = urllib.urlopen(url)
    arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
    image = cv2.imdecode(arr, -1)
    return image


class GenderDetection:
    def __init__(self):

        self.FACE_PROTO = "./models/opencv_face_detector.pbtxt"
        self.FACE_MODEL = "./models/opencv_face_detector_uint8.pb"

        self.AGE_PROTO = "./models/age_deploy.prototxt"
        self.AGE_MODEL = "./models/age_net.caffemodel"

        self.GENDER_PROTO = "./models/gender_deploy.prototxt"
        self.GENDER_MODEL = "./models/gender_net.caffemodel"
        self.result = []

        self.FACE_NET = cv2.dnn.readNet(self.FACE_MODEL, self.FACE_PROTO)
        self.AGE_NET = cv2.dnn.readNet(self.AGE_MODEL, self.AGE_PROTO)
        self.GENDER_NET = cv2.dnn.readNet(self.GENDER_MODEL, self.GENDER_PROTO)

        self.MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
        self.AGE_LIST = ["(0-2)", "(4-6)", "(8-12)", "(15-20)", "(25-32)", "(38-43)", "(48-53)", "(60-100)"]
        self.GENDER_LIST = ["Male", "Female"]

        self.box_padding = 20

    def get_face_box(self, net, frame, conf_threshold=0.7):
        frame_copy = frame.copy()
        frame_height = frame_copy.shape[0]
        frame_width = frame_copy.shape[1]
        blob = cv2.dnn.blobFromImage(frame_copy, 1.0, (300, 300), [104, 117, 123], True, False)

        net.setInput(blob)
        detections = net.forward()
        boxes = []

        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf_threshold:
                x1 = int(detections[0, 0, i, 3] * frame_width)
                y1 = int(detections[0, 0, i, 4] * frame_height)
                x2 = int(detections[0, 0, i, 5] * frame_width)
                y2 = int(detections[0, 0, i, 6] * frame_height)
                boxes.append([x1, y1, x2, y2])
                cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), int(round(frame_height / 150)), 8)

        return frame_copy, boxes

    def age_gender_detector(self, url):
        image = read_image_from_url(url)
        resized_image = cv2.resize(image, (640, 480))

        frame = resized_image.copy()
        frame_face, boxes = self.get_face_box(self.FACE_NET, frame)

        for box in boxes:
            face = frame[max(0, box[1] - self.box_padding):min(box[3] + self.box_padding, frame.shape[0] - 1), \
                   max(0, box[0] - self.box_padding):min(box[2] + self.box_padding, frame.shape[1] - 1)]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), self.MODEL_MEAN_VALUES, swapRB=False)
            self.GENDER_NET.setInput(blob)
            gender_predictions = self.GENDER_NET.forward()
            gender = self.GENDER_LIST[gender_predictions[0].argmax()]
            gender_res = "Gender: {}".format(gender)

            print(gender_res)
            self.AGE_NET.setInput(blob)
            age_predictions = self.AGE_NET.forward()
            age = self.AGE_LIST[age_predictions[0].argmax()]
            age_res = "Age: {}, conf: {:.3f}".format(age, age_predictions[0].max())
            print(age_res)
            result_dict = {
                "gender": gender_res
            }
            self.result.append(result_dict)
        return self.result


if __name__ == "__main__":
    gender_detection_model = GenderDetection()
    res = gender_detection_model.age_gender_detector('https://i.pinimg.com/564x/3f/64/2f/3f642fe948405e9b2eb39094ec3be372.jpg')
    print(res)