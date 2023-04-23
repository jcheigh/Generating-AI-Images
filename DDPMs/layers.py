'''this python defines custom layers used in U-Net'''

# generic libraries
import math
from einops import rearrange

# keras and tensorflow
import tensorflow as tf
from tensorflow import einsum
from keras import Sequential
from keras.layers import Layer, Dense, Conv2D, Softmax
import tensorflow_addons as tfa

gpu_device = tf.test.gpu_device_name()
cpu_device = '/cpu:0'
# set CPU the device for now
device = gpu_device

'''
    Sinusoidal positional embedding layer: this layer generates a embeddings using sin and cos functions and
    allows our model to learn the relative positions of input sequential data. In our specific case, we use 
    this layer to transfer time steps to time ecodings in time embedding space
'''
class SinusoidalPosEmb(Layer):
    def __init__(self, dim, max_positions=10000):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim
        self.max_positions = max_positions

    def call(self, x, training=True):
        with tf.device(device):
            x = tf.cast(x, tf.float32)
            half_dim = self.dim // 2
            emb = math.log(self.max_positions) / (half_dim - 1)
            emb = tf.exp(tf.range(half_dim, dtype=tf.float32) * -emb)
            emb = x[:, None] * emb[None, :]
            emb = tf.concat([tf.sin(emb), tf.cos(emb)], axis=-1)
            return emb

'''
    Identity layer: this layer returns the tensor of the same shape and content as the input
'''
class Identity(Layer):
    def __init__(self):
        super(Identity, self).__init__()

    def call(self, x, training=True):
        with tf.device(device):
            return tf.identity(x)


'''
    Residual Layer: this layer takes in some function and apply the function on an input. In our case,
    it will be used for incorporating attention.
'''
class Residual(Layer):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def call(self, x, training=True):
        with tf.device(device):
            return self.fn(x, training=training) + x

'''
    Normalization layer: this layer normalizes the input with respect to the mean and variance
'''
class LayerNorm(Layer):
    def __init__(self, dim, eps=1e-5, **kwargs):
        super(LayerNorm, self).__init__(**kwargs)
        self.eps = eps
        self.g = tf.Variable(tf.ones([1, 1, 1, dim]))
        self.b = tf.Variable(tf.zeros([1, 1, 1, dim]))

    def call(self, x, training=True):
        with tf.device(device):
            var = tf.math.reduce_variance(x, axis=-1, keepdims=True)
            mean = tf.reduce_mean(x, axis=-1, keepdims=True)
            x = (x - mean) / tf.sqrt((var + self.eps)) * self.g + self.b
            return x

'''
    Pre-Normalization layer: this layer calls the normalization layer of a certain dimension
    to normalize the input and apply a specified function afterwards.
'''
class PreNorm(Layer):
    def __init__(self, dim, fn):
        super(PreNorm, self).__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def call(self, x, training=True):
        with tf.device(device):
            x = self.norm(x)
            return self.fn(x)

'''
    Sigmoid Linear Unit (SiLU) layer: this layer defines the sigmoid linear function given by
    silu(x) = x * sigmoid(x). It is useful due to its self-stabilizing property.
'''
def silu(x):
    with tf.device(device):
        return x * tf.nn.sigmoid(x)

class SiLU(Layer):
    def __init__(self):
        super(SiLU, self).__init__()

    def call(self, x, training=True):
        with tf.device(device):
            return silu(x)

'''
    Gaussian Error Linear Unit layer: this layer defines Gaussian error linear function given by
    gelu(x) = x * sigma(x), where sigma is the culmulative distribution function for Gaussian. It is
    useful in bridging stochatic regularizers with non-liniearities.
'''
def gelu(x, approximate=False):
    with tf.device(device):
        if approximate:
            coeff = tf.cast(0.044715, x.dtype)
            return 0.5 * x * (1.0 + tf.tanh(0.7978845608028654 * (x + coeff * tf.pow(x, 3))))
        else:
            return 0.5 * x * (1.0 + tf.math.erf(x / tf.cast(1.4142135623730951, x.dtype)))

class GELU(Layer):
    def __init__(self, approximate=False):
        super(GELU, self).__init__()
        self.approximate = approximate

    def call(self, x, training=True):
        with tf.device(device):
            return gelu(x, self.approximate)

'''
    Block layer: this layer defines a unit block in the following Residual networ layer
'''
class Block(Layer):
    def __init__(self, dim, groups=8):
        super(Block, self).__init__()
        self.proj = Conv2D(dim, kernel_size=3, strides=1, padding='SAME')
        self.norm = tfa.layers.GroupNormalization(groups, epsilon=1e-05)
        self.act = SiLU()


    def call(self, x, gamma_beta=None, training=True):
        with tf.device(device):
            x = self.proj(x)
            x = self.norm(x, training=training)

            if gamma_beta is not None:
                gamma, beta = gamma_beta
                x = x * (gamma + 1) + beta

            x = self.act(x)
            return x

'''
    Residual Network layer: this layer defines the layer of our residual network. This allows us to 
    to skip layers without affecting performance, which in turn enables our U-Net to learn features at 
    different stages of segmentation.
'''

class ResnetBlock(Layer):
    def __init__(self, dim, dim_out, time_emb_dim=None, groups=8):
        super(ResnetBlock, self).__init__()

        self.mlp = Sequential([
            SiLU(),
            Dense(units=dim_out * 2)
        ]) if time_emb_dim is not None else None

        self.block1 = Block(dim_out, groups=groups)
        self.block2 = Block(dim_out, groups=groups)
        self.res_conv = Conv2D(filters=dim_out, kernel_size=1, strides=1) if dim != dim_out else Identity()

    def call(self, x, time_emb=None, training=True):
        with tf.device(device):
            gamma_beta = None
            if self.mlp is not None and time_emb is not None:
                time_emb = self.mlp(time_emb)
                time_emb = rearrange(time_emb, 'b c -> b 1 1 c')
                gamma_beta = tf.split(time_emb, num_or_size_splits=2, axis=-1)

            h = self.block1(x, gamma_beta=gamma_beta, training=training)
            h = self.block2(h, training=training)

            return h + self.res_conv(x)

'''
    Linear Attention layer: this layer defines alinear attension layer. Unlike attension layer, 
    it takes in a constant-sized input. Each output activation in a linear layer is a linear 
    combination of the activations in the previous layer. It allows the combination between 
    attention modules and neural networks more flexible and versatile. 
'''
class LinearAttention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads
        self.attend = Softmax()
        self.to_qkv = Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)

        self.to_out = Sequential([
            Conv2D(filters=dim, kernel_size=1, strides=1),
            LayerNorm(dim)
        ])

    def call(self, x, training=True):
        with tf.device(device):
            b, h, w, c = x.shape
            qkv = self.to_qkv(x)
            qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
            q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)

            q = tf.nn.softmax(q, axis=-2)
            k = tf.nn.softmax(k, axis=-1)

            q = q * self.scale
            context = einsum('b h d n, b h e n -> b h d e', k, v)

            out = einsum('b h d e, b h d n -> b h e n', context, q)
            out = rearrange(out, 'b h c (x y) -> b x y (h c)', h=self.heads, x=h, y=w)
            out = self.to_out(out, training=training)
            return out

'''
    Attension layer: this layer defines regular attension later. This performs a weighted mean reduction
    and allows us to dynamically highlight relevant features of the sequential input data (weighing
    the importance of each part of the input).
'''
class Attention(Layer):
    def __init__(self, dim, heads=4, dim_head=32):
        super(Attention, self).__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.hidden_dim = dim_head * heads

        self.to_qkv = Conv2D(filters=self.hidden_dim * 3, kernel_size=1, strides=1, use_bias=False)
        self.to_out = Conv2D(filters=dim, kernel_size=1, strides=1)

    def call(self, x, training=True):
        with tf.device(device):
            b, h, w, c = x.shape
            qkv = self.to_qkv(x)
            qkv = tf.split(qkv, num_or_size_splits=3, axis=-1)
            q, k, v = map(lambda t: rearrange(t, 'b x y (h c) -> b h c (x y)', h=self.heads), qkv)
            q = q * self.scale

            sim = einsum('b h d i, b h d j -> b h i j', q, k)
            sim_max = tf.stop_gradient(tf.expand_dims(tf.argmax(sim, axis=-1), axis=-1))
            sim_max = tf.cast(sim_max, tf.float32)
            sim = sim - sim_max
            attn = tf.nn.softmax(sim, axis=-1)

            out = einsum('b h i j, b h d j -> b h i d', attn, v)
            out = rearrange(out, 'b h (x y) d -> b x y (h d)', x = h, y = w)
            out = self.to_out(out, training=training)

            return out