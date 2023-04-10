import utils
import model 
import tensorflow as tf

# config hyperprameters
epochs = 10
batch_size = 32
latent_dim = 2

# gpu
gpu_device = tf.test.gpu_device_name()
cpu_device = '/cpu:0'

def main():
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
    model.train(model=vae, device=cpu_device, epochs=epochs, x=x_train, x_test=x_test)

main()