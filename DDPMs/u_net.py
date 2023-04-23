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
device = gpu_device

class Unet(Model):
    def __init__(self, dim=64, init_dim=None, out_dim=None, dim_mults=(1,2,4,8), channels=1, resnet_block_groups=8, learned_variance=False, sinusoidal_cond_mlp=True):
        super(Unet, self).__init__()
        
        # determine dimensions and other configurations
        self.channels = channels
        d = dim // 3 * 2
        if init_dim is not None:
            init_dim = init_dim
        else:
            init_dim = d() if isfunction(d) else d
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        
        # time embeddings
        time_dim = dim * 4
        self.sinusoidal_cond_mlp = sinusoidal_cond_mlp
        
        self.time_mlp = Sequential([
            SinusoidalPosEmb(dim),
            Dense(units=time_dim),
            GELU(),
            Dense(units=time_dim)
        ], name="time embeddings")
        
        # layers
        self.downs = []
        self.ups = []
        num_resolutions = len(in_out)

        self.init_conv = Conv2D(filters=init_dim, kernel_size=7, strides=1, padding='same')
        block_klass = partial(ResnetBlock, groups = resnet_block_groups)
        
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append([
                block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                Conv2D(filters=dim_out, kernel_size =4, strides=2, padding='same') if not is_last else Identity()
            ])
  
        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append([
                block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                Conv2DTranspose(filters=dim_in, kernel_size=4, strides=2, padding='same') if not is_last else Identity()
            ])
        
        default_out_dim = channels * (1 if not learned_variance else 2)
        
        if out_dim is not None:
            self.out_dim = out_dim
        else:
            self.out_dim = default_out_dim() if isfunction(default_out_dim) else default_out_dim
        
        self.final_conv = Sequential([
            block_klass(dim * 2, dim),
            Conv2D(filters=self.out_dim, kernel_size=1, strides=1)
        ], name="output")
        
    def call(self, x, time=None, training=True, **kwargs):
        with tf.device(device):
            x = self.init_conv(x)
            t = self.time_mlp(time)
            h = []

            for block1, block2, attn, downsample in self.downs:
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                h.append(x)
                x = downsample(x)

            x = self.mid_block1(x, t)
            x = self.mid_attn(x)
            x = self.mid_block2(x, t)

            for block1, block2, attn, upsample in self.ups:
                x = tf.concat([x, h.pop()], axis=-1)
                x = block1(x, t)
                x = block2(x, t)
                x = attn(x)
                x = upsample(x)

            x = tf.concat([x, h.pop()], axis=-1)
            x = self.final_conv(x)
            return x