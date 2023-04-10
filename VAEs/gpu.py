import tensorflow as tf
import model 
import utils

gpu_device = tf.test.gpu_device_name()
cpu_device = '/cpu:0'

'''Test and get GPU'''
def check_gpu():
    import timeit, functools
    print('CPU (s):')
    cpu_time = timeit.Timer(functools.partial(run_cpu, '/cpu:0'))
    print(cpu_time.timeit(1000))
    print('GPU (s):')
    gpu_time = timeit.Timer(functools.partial(run_gpu, tf.test.gpu_device_name()))
    print(gpu_time.timeit(1000))
    
x_train, _ = utils.load_data()
x_train = utils.preprocess_image_data(x_train)
x_train = utils.split_batch(image_data=x_train, batch_size=32)

net = model.VAE(latent_dim=2)

def run_cpu(cpu_device):
    with tf.device(cpu_device):
        model.train_per_batch(net, x_train, cpu_device)

def run_gpu(gpu_device):
    with tf.device(gpu_device):
        model.train_per_batch(net, x_train, gpu_device)

check_gpu()
