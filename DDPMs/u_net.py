from layers import SinusoidalPosEmb, Identity, Residual, PreNorm, GELU
from layers import ResnetBlock, LinearAttention, Attention

import tensorflow as tf
from keras.layers import Conv2D, Conv2DTranspose, Dense
from keras import Model, Sequential
from functools import partial
from inspect import isfunction

gpu_device = tf.test.gpu_device_name()
cpu_device = '/cpu:0'
# set CPU the device for now
device = cpu_device

class Unet(Model):
    def __init__(self, dim=64, channels=1):
        super(Unet, self).__init__()
        # number of channels
        self.channels = channels
        
        # determine dimensions
        init_dim = dim//3*2
        dim_mults=(1,2,4,8)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # define front convolution 
        self.init_conv = Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='same')

        # define multi-layer perceptron for time embedding
        time_dim = dim * 4
        self.time_mlp = Sequential([
            SinusoidalPosEmb(dim),
            Dense(units=time_dim),
            GELU(),
            Dense(units=time_dim)
        ])

        # define the final resolution of encoder (= the input resolution of decoder)
        num_resolutions = len(in_out)

        # define ResnetBlock layer with 8 groups 
        resnet_block = partial(ResnetBlock, groups=8)

        # define down-sampling conv-mlp (encoder of U-net)
        self.downs = []
        for level, (dim_in, dim_out) in enumerate(in_out):
            # create a new down-sampling layer
            self.downs.append([
                resnet_block(dim_in, dim_out, time_emb_dim=time_dim),
                resnet_block(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Conv2D(filters=dim_out, kernel_size =4, strides=2, padding='same') if level < (num_resolutions - 1) else Identity()
            ])

        # bottle-neck of U-net with skip connection
        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = resnet_block(mid_dim, mid_dim, time_emb_dim=time_dim)

        # define up-sampling conv-mlp (decoder of U-net)
        self.ups = []
        for level, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            # create a new up-sampling level
            self.ups.append([
                resnet_block(dim_out * 2, dim_in, time_emb_dim=time_dim),
                resnet_block(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Conv2DTranspose(filters=dim_in, kernel_size=4, strides=2, padding='same') if level < (num_resolutions - 1) else Identity()
            ])
        
        # define back convolution
        self.out_dim = channels
        self.final_conv = Sequential([
            resnet_block(dim * 2, dim),
            Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ])
        
    def call(self, x, time, training=True, **kwargs):
        with tf.device(device):
            # front conv
            x = self.init_conv(x)

            # time embedding
            t = self.time_mlp(time)

            # move down the encoder
            h = []
            for down_block1, down_block2, attention, downsample in self.downs:
                x = down_block1(x, t)
                x = down_block2(x, t)
                x = attention(x)
                h.append(x)
                x = downsample(x)

            # bottleneck
            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            # move up the decoder
            for up_block1, up_block2, attention, upsample in self.ups:
                x = tf.concat([x, h.pop()], axis=-1)
                x = up_block1(x, t)
                x = up_block2(x, t)
                x = attention(x)
                x = upsample(x)
            x = tf.concat([x, h.pop()], axis=-1)

            # back conv
            x = self.final_conv(x)
            return x