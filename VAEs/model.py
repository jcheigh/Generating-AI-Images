'''
    Convolutional VAEs based on the tensorflow 
    tutorial: https://www.tensorflow.org/tutorials/generative/cvae
    Slight modifications were made to further improve the model
'''

import tensorflow as tf
from tensorflow import keras, random, exp, split
from keras import Model, Sequential
from keras.layers import InputLayer, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten

class VAE(Model):
    '''Basic Convolutional VAE'''

    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        self.encoder = self.generate_encoder()
        self.decoder = self.generate_decoder()

    def generate_encoder(self):
        return Sequential([
            InputLayer(input_shape=(28, 28, 1)),
            Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu'),
            Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu'),
            Flatten(),
            # because the output of encoder has two parts, the mean and 
            # log-variance of the posterior distribution q(z|x)
            Dense(self.latent_dim * 2)
        ])
    
    def generate_decoder(self):
        return Sequential([
            # the input to decoder is z sampled from the Gaussian (through 
            # reparameterization trick)
            InputLayer(input_shape=(self.latent_dim,)),
            Dense(units=32*7*7, activation='relu'),
            Reshape(target_shape=(7,7,32)),
            Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', activation='relu'),
            Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', activation='relu'),
            # no activation as this is just for the final output image (i.e., x)
            Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same', activation=None),
        ])

    def encode(self, x):
        mean, logvar = split(self.encoder(x), num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def sample_by_reparameterization(self, mean, logvar):
        epsilon = random.normal(shape=mean.shape)
        z = mean + epsilon * exp(logvar * 0.5)
        return z

    def decode(self, z):
        x_hat = self.decode(z)
        return x_hat