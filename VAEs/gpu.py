import tensorflow as tf
import model 
import utils

gpu_device = tf.test.gpu_device_name()
cpu_device = '/cpu:0'

'''Test and get GPU'''
def check_gpu():
    import timeit, functools
    print('CPU (s):')
    cpu_time = timeit.Timer(functools.partial(run_cpu, cpu_device))
    print(cpu_time.timeit(1))
    print('GPU (s):')
    gpu_time = timeit.Timer(functools.partial(run_gpu, gpu_device))
    print(gpu_time.timeit(1))

# def run_cpu():
#     with tf.device('/cpu:0'):
#         random_image_gpu = tf.random.normal((100, 100, 100, 3))
#         net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
#         tf.math.reduce_sum(net_gpu)

# def run_gpu():
#     with tf.device(device_name):
#         random_image_gpu = tf.random.normal((100, 100, 100, 3))
#         net_gpu = tf.keras.layers.Conv2D(32, 7)(random_image_gpu)
#         tf.math.reduce_sum(net_gpu)
    
x_train, _ = utils.load_data()
x_train = utils.preprocess_image_data(x_train)
x_train = utils.split_batch(image_data=x_train, batch_size=32)

net = model.VAE(latent_dim=2)

def run_cpu(cpu_device):
    with tf.device(cpu_device):
        for x_batch in x_train:
            model.train_per_batch(net, x_batch, cpu_device)

def run_gpu(gpu_device):
    with tf.device(gpu_device):
        for x_batch in x_train:
            model.train_per_batch(net, x_batch, gpu_device)
