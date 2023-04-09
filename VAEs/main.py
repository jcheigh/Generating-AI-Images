import utils
import model 

def main():
    # get data
    x_train, x_test = utils.load_data()

    # preprocess data
    x_train = utils.preprocess_image_data(x_train)
    x_test = utils.preprocess_image_data(x_test)

    # split data into batches
    x_train = utils.split_batch(image_data=x_train, batch_size=32)
    x_test = utils.split_batch(image_data=x_test, batch_size=32)
    #print(np.shape(list(x_train)))

    vae = model.VAE(latent_dim=2)

main()