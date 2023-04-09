'''
    Convolutional VAEs based on the tensorflow 
    tutorial: https://www.tensorflow.org/tutorials/generative/cvae
    Slight modifications were made to further improve the model
'''

import numpy as np
from math import sqrt
from tensorflow import data
from keras import datasets

def load_data():
    (x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
    return x_train, x_test

def preprocess_image_data(data):
    # reshape, normalize, and binarize (gray) data
    shape = data.shape
    image_data = data.reshape((shape[0], shape[1], shape[2], 1)) / 255.
    return np.where(image_data < 0.5, 0.0, 1.0).astype('float32')

def split_batch(image_data, batch_size):
    data_size = len(image_data)
    return (data.Dataset.from_tensor_slices(image_data).shuffle(data_size).batch(batch_size))