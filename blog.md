# Generating MNIST Digits with VAE, DCGAN, and Diffusion Models in Keras 

## Insert Introduction

# DCGAN Section

We'll now turn to GANs and DCGANs! After reading this section, you will learn:
1. The theory and intuition behind GANs
2. How to implement DCGANs in Keras
3. The applications and limitations of GANs/DCGANs

Ok, let's get started. The objective of Generative Adversarial Networks (GANs) is to *generate new data* similar to that of the initial data distribution fed into the model. For example, we will be generating images of handwritten digits in the style of the MNIST dataset. 

So, how do GANs learn to do this? Well, by playing a game! In GANs, there are two neural networks, namely the **generator** G and the **discriminator** D. The generator creates new data, whereas the discriminator takes this new data and classifies it as real or fake. These networks are trained against each other and thus are *adversarial*. The generator's objective is to "beat" the discriminator; it tries to generate data that the discriminator is unable to differentiate from real data. The discriminator's objective is to beat the generator; it tries to differentiate (or *discriminate*)between the fake generated data from the real data.

Both the generator and discriminator are crucial to the success of GANs, as they both improve from being pitted against each other. The generator takes the feedback from the discriminator and learns how to better fool the discriminator. In other words, it learns how to generate data more similar to the initial data distribution. The discriminator, on the other hand, learns how to better differentiate between real and fake images. By doing it so it forces the generator to improve (else it will not be able to fool the discriminator). 

So, we get this interesting training process. The discriminator and generator both start off untrained. As they "play" against each other more and more, both models improve until the point where the generator is *really* good at generating new data, and the discriminator is *really* good at discriminating between real and fake data. At this point, we're done! We can just use the generator, which is now capable of generating new data similar to that of the initial data distribution. 

Ok, now let's dive more into the math. We are given some initial data distribution $p_{\text{data}}$. Our goal is to learn some distribution $p_g$ over data $\textbf{x}$ (i.e. the same shape as the initial data distribution) such that $p_{\text{data}}$ and $p_g$ are as similar as possible. We begin by defining a prior on input noise variables $p_z(\textbf{z})$. In practice, $p_z(\textbf{z})$ is usually either a Gaussian (normal) distribution or a uniform distribution. 

It's important to understand what the noise represents and why we start with it. Some $z \sim p_z(\textbf{z})$ represent the *latent features* of the generated data. Latent features basically are the hidden features of the image not directly observable. For example, maybe $z$ represents the hair color, face shape, and more if we were generating images of faces. By starting with noise, we also will get more diverse data samples, as the model won't be able to just memorize the data samples from the initial data distribution. This helps prevent overfitting. Further, it allows for more variety in the generated data. For example, it prevents the model from just generating the same digit in the same way each time. 


The generator takes the noise $z \sim p_z(\textbf{z})$ and outputs $x = G(z)$. Notice that the generator defines the distribution $p_g$. The discriminator takes the generated data $x$ and outputs $D(x) \in [0,1]$, where $D(x)$ represents the probability that $x$ came from the initial data rather than from $p_g$. 


The generator takes $z$ and outputs $x = G(z)$. The discriminator then takes $x$ and outputs $P[\text{real} \ |\ x] = D(x)$ (i.e. the probability that $x$ is a real datapoint sampled from $p_{\text{data}}$). We also feed the discriminator some real data, so it actually learns to differentiate between real and fake data. In total this describes the feed forward aspect of the model, so it's not time to think about training and backpropagation.  

Since the generator makes the first step, the discriminator is the later in the feed forward process and thus will be earlier in the training process. The discriminator is a binary classifier, so we will be using a version of binary cross entropy:
$$
\max_D V(D) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))].
$$

Ok, let's break down this equation. The goal of the discriminator is to classify real data as real and fake data as fake. The first term of this expression encourages the discriminator to classify real data as $1$ i.e. real. The second the encourages the discriminator to classify the fake data as $0$ i.e. fake. On the generator side things are pretty analogous, with the key difference being the generator wants to fool the discriminator.
$$
\min_G V(G) = \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))].
$$

Often we combine these two into a single *minimax* game:
$$
\min_G \max_D V(D, G) = \mathbb{E}_{\mathbf{x} \sim p_{\text{data}}(\mathbf{x})}[\log D(\mathbf{x})] + \mathbb{E}_{\mathbf{z} \sim p_z(\mathbf{z})}[\log(1 - D(G(\mathbf{z})))].
$$

Here, the weights of the dicriminator are updated first, so it is as if the discriminator "moves" first in this game. The discriminator knows that the generator will have a chance to update its weights right afterwards, so, like in any minimax game, the discriminator *moves in a way to maximize its utility given the generator will act optimally*. In other words, the discriminator assumes that the generator will make the best possible move, and it makes a move to minimize its own personal loss. 

And that's it! We've now covered the theory and intuition behind GANs. The natural question now is: where and why do GANs have trouble? We'll now go over a few common issues with GANs. GANs are very difficult to train. This instability relies from how dependent the generator and discriminator are on each other. If one gets much stronger than the other, training may begin the fail. Another issue is *mode collapse*. "If it ain't broke don't fix it". Formally, mode collapse is when the generator only produces a small subset of the potential data. One example is if it only produces a few specific data samples everytime since the discriminator for whatever reason has trouble with it. Finally, another problem with virtually all generative AI is evaluation: there is no standard way to evaluate the quality of the generated samples.

Now we will discuss DCGAN. Since GANs typically are used for generating images, it intuitively makes sense to have some sort of convolutional layers in the model. This is exactly what DCGAN does. Below are the notes from the original paper on DCGAN for turning a GAN to a DCGAN:
- Replace any pooling layers with strided convolutions (discriminator) and fractional-strided convolutions (generator). 
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

It's important to understand why we use different type of convolutional layers for the generator and discriminator. The discriminator needs to take an image, down-sample it to learn latent features, and then classify based on these features. The fact that the discriminator is downsampling implies we should use standard convolutional layers (here suggested with stride as well). On the other hand, the generator is almost doing the opposite! It starts with noise $\mathbf{z}$ (recall the noise represents the latent features of the image it will generate!) and up-samples to create an image. So, rather than use standard convolutional layers to downsample, we use *de*convolutional layers (i.e. transposed convolutions) to upsample. 

Great! we've covered the theory of both GANs and DCGANs, so let's see how we can implement a DCGAN to generate imaes in the style of MNIST digits. 