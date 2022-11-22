## _lookAt_


# Requirements

 - Tensorflow v2.0 or upper
 - Python v3.8 or upper


# Description 

LookAt is based on two Tensorflow models, ``Fashion-MNIST`` and ``YOLOv3``, which was created and trained to predict clothes on images.
[```Fashion-MNIST```](https://www.tensorflow.org/tutorials/keras/classification)  has 10 clothes classes

- 0: T-shirt/top
- 1: Trouser
- 2: Pullover
- 3: Dress
- 4: Coat
- 5: Sandal
- 6: Shirt
- 7: Sneaker
- 8: Bag
- 9: Ankle boot
- 
App predicts cropped images based on these 10 classes.
Model has 60.000 train images and 10.000 test images. 

![fashion-mnist-sprite](https://user-images.githubusercontent.com/92037197/196115022-14aa8275-ab5f-4502-a19f-c9f8ec565dca.png)

All images are represented in ```28x28x1``` shape with gray color, so for predict images input images should be in the same shape and color

![Plot-of-a-Subset-of-Images-from-the-Fashion-MNIST-Dataset-1024x768](https://user-images.githubusercontent.com/92037197/196117297-14687a16-839d-4a89-b1eb-58772be73472.png)


## YOLOv3

[```YOLOv3```](https://github.com/pythonlessons/TensorFlow-2.x-YOLOv3) a basic object detection model, but we use it to detect clothes in images and crop them. 
![city_pred](https://user-images.githubusercontent.com/92037197/196117650-4ebd6804-31f8-413c-a442-f2f470ce6a62.jpg)
![results](https://user-images.githubusercontent.com/92037197/196117699-73e29b06-30cc-4a79-8c25-027c20fd498f.jpg)
## API

API created using [```Fast-API```](https://fastapi.tiangolo.com/) python library. Users upload images on API, models start to predict.

![Screenshot (12)](https://user-images.githubusercontent.com/92037197/196118021-70b5222e-a183-4480-8ab5-9f9dae6e3c2e.png)
API has two requests -  ```/upload``` and ```/predict/{id}```.


## /Upload 
method  is a ```POST``` request and requires an image. This method uses the ```YOLOv3``` model, and returns crop images by unique id.
## /Predict/{id} 
methode is ```GET``` request and use ```Fashion-MNIST```, its return list objects(clothes in JSON) in images, this method has required parameter - ``id``, which we get from previous method

