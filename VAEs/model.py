'''
    Convolutional VAEs based on the following references:
    https://www.tensorflow.org/tutorials/generative/cvae
    https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73
    https://arxiv.org/pdf/1907.08956.pdf

    Slight modifications were made to further improve the model
'''

import gpu

import numpy as np
import time
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras, random, nn
from keras import Model, Sequential, optimizers, metrics
from keras.layers import InputLayer, Dense, Conv2D, Conv2DTranspose, Reshape, Flatten

'''Test and get GPU'''
# test and get Colab gpu
device_name = gpu.test_gpu()
def check_gpu():
    import timeit
    print('CPU (s):')
    cpu_time = timeit.timeit('run_cpu()', number=10)
    print(cpu_time)
    print('GPU (s):')
    gpu_time = timeit.timeit('run_gpu()', number=10)
    print(gpu_time)
    print('GPU speedup over CPU: {}x'.format(int(cpu_time/gpu_time)))

def run_cpu():
    cpu_start = time.time()
    with tf.device('/cpu:0'):
        random_image_cpu = tf.random.normal((100, 100, 100, 3))
        net_cpu = tf.keras.layers.Conv2D(32, 7)(random_image_cpu)
        tf.math.reduce_sum(net_cpu)
    cpu_end = time.time()

def run_gpu():
    gpu_start = time.time()
    with tf.device(device_name):
        random_image_gpu = tf.random.normal((100, 100, 100, 3))
        net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
        tf.math.reduce_sum(net_gpu)
    gpu_end = time.time()

'''Basic Convolutional VAE'''
class VAE(Model):

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
            Conv2DTranspose(filters=1, kernel_size=3, strides=1, padding='same')
        ])

    def encode(self, x):
        with tf.device(device_name=device_name):
            mean, logvar = tf.split(self.encoder(x), num_or_size_splits=2, axis=1)
            return mean, logvar
    
    def reparameterize(self, mean, logvar):
        with tf.device(device_name=device_name):
            e = random.normal(shape=mean.shape)
            z = mean + e * tf.exp(logvar * 0.5)
            return z

    def decode(self, z):
        with tf.device(device_name=device_name):
            xhat = self.decoder(z)
            return xhat
    
    '''used during inference'''
    def sample(self, rand_vec=None):
        with tf.device(device_name=device_name):
            if rand_vec == None:
                rand_vec = random.normal(shape=(100, self.latent_dim))
            return tf.sigmoid(self.decode(rand_vec)) # binarize the result for visibility


'''Define loss function and training proceedures'''

def log_normal(sample, mean, logvar):
    with tf.device(device_name=device_name):
        prob = -0.5 * ((sample - mean) ** 2.0 * tf.exp(-logvar) + logvar + tf.math.log(2.0 * np.pi))
        return tf.reduce_sum(prob, axis=1)

def get_loss(model, x):
    with tf.device(device_name=device_name):
        mean, logvar = model.encode(x) # defining q(z|x)
        z = model.reparameterize(mean, logvar) # sampled from q(z|x)
        xhat = model.decode(z) # generated by obtaining p(x|z)
        prob = nn.sigmoid_cross_entropy_with_logits(labels=x, logits=xhat)

        logpxz = -tf.reduce_sum(prob, axis=[1,2,3]) # just reconstruction error
        logpz = log_normal(z, mean, logvar) # standard Gaussian (assumed)
        logqzx = log_normal(z, mean, logvar) # Gaussian obtained through encoder

        # return the average value for each sample within this batch
        return -tf.reduce_mean(logpxz + logpz - logqzx)

'''use Adan'''
def train_per_batch(model, x):
    optimizer = optimizers.Adam(learning_rate=1e-4)
    with tf.device(device_name=device_name):
        with tf.GradientTape() as gt:
            loss = get_loss(model, x)
            gradient = gt.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradient, model.trainable_variables))
        return loss

def train(model, epochs, x, x_test=None):
    # for each epoch
    with tf.device(device_name=device_name):
        for epoch in range(epochs):
            start_time = time.time()
            train_loss = metrics.Mean()
            for x_batch in x:
                train_loss(train_per_batch(model, x_batch))
            time_elapsed = time.time() - start_time
            loss = train_loss.result()
            print(f'Epoch: {epoch}, train loss: {loss}, time elapsed: {time_elapsed}')

            if x_test is not None:
                for test_batch in x_test:
                    x_sample = test_batch[0:16,:,:,:]
                    break
                pred = predict(model=model, x=x_sample)
                show_pred(pred, epoch)

def predict(model, x):
    with tf.device(device_name=device_name):
        mean, logvar = model.encode(x) 
        z = model.reparameterize(mean, logvar) 
        return model.sample(z) 

def show_pred(pred, epoch=None):
    plt.figure(figsize=(4,4))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.imshow(pred[i, :, :, 0], cmap='gray')
        plt.axis('off')
    if epoch is not None:
        plt.savefig(f'./digits_images/epoch{epoch}.png')
    else:
        plt.show()