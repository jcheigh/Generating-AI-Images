import tensorflow as tf

import utils
import model 
import gpu

# config hyperprameters
epochs = 10
batch_size = 32
latent_dim = 2


def main():
    # test and get Colab gpu
    device_name = gpu.test_gpu()

    # get data
    x_train, x_test = utils.load_data()

    # preprocess data
    x_train = utils.preprocess_image_data(x_train)
    x_test = utils.preprocess_image_data(x_test)

    # split data into batches
    x_train = utils.split_batch(image_data=x_train, batch_size=batch_size)
    x_test = utils.split_batch(image_data=x_test, batch_size=batch_size)

    # define and train model
    vae = model.VAE(latent_dim=latent_dim)
    model.train(model=vae, device_name=device_name, epochs=epochs, x=x_train, x_test=x_test)

main()