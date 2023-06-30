## _lookAt_

  [![Tensorflow logo](https://www.gstatic.com/devrel-devsite/prod/v1ba1082cb0bd9b151fb2d708d31f382e850c5d60b82de6be21570706ce15859e/tensorflow/images/lockup.svg)](https://www.tensorflow.org/)[![Python](https://www.python.org/static/community_logos/python-powered-w-100x40.png)](https://www.python.org/) 

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

API created using [```Fast-API```](https://fastapi.tiangolo.com/) python framework. Users upload images on API, models start to predict.

![Screenshot (48)](https://user-images.githubusercontent.com/92037197/216965340-9aa7b70a-6498-4351-b032-2d9222bf500b.png)
API has two parameters in body  -  ```user_id``` and ```url```.


## /api/load_image/
method  is a ```POST```. This method uses the ```Fashion-MNIST``` model for non-person images, and  ```YOLOv3``` model for images with persons, and returns predicted images with dominant color.




![Screenshot (49)](https://user-images.githubusercontent.com/92037197/216965398-9fd68670-242a-4d5a-8049-48862b34bb48.png)

