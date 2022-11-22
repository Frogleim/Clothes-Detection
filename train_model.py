import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
import time

fashion_mnist = tf.keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
train_images = train_images.astype('float32')
test_images = test_images.astype('float32')

train_images = train_images / 255.0

test_images = test_images / 255.0


def create_model():
    model = tf.keras.Sequential()
    model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation=tf.nn.relu))
    model.add(Dropout(0.2))
    model.add(Dense(10, activation=tf.nn.softmax))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


build_model = create_model()


def train_model(model):
    model.fit(x=train_images, y=train_labels, epochs=1000)

    return model


if __name__ == '__main__':

    while True:
        ready_model = train_model(build_model)
        ready_model.save('./model/')
        time.sleep(3600*4)