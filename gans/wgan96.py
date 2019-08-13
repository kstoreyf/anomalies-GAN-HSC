# *********************************************************************************
# * File Name : wgan96.py
# * Creation Date : 2019-07-26
# * Created By : kstoreyf
# * Description : Implemtation of a WGAN-GP to generate 
# *     images from 96x96 galaxy cutouts. Based on
# *     https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py.
# *     Only tested on wgan-gp mode.
# *********************************************************************************

import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot
import tflib.datautils


MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 32 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 50000 # How many generator iterations to train for
SAMPLE_ITERS = 100 # Multiples at which to generate image sample
SAVE_ITERS = 50000
NSIDE = 96 # Don't change this without changing the model layers!
OUTPUT_DIM = NSIDE*NSIDE # Number of pixels in MNIST (28*28)

#tag = 'i85k_96x96_norm'
#imarr_fn = "../data/imarrs_np/hsc_{}.npy".format(tag)
tag = 'i20.0_norm'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'

out_dir = f'../training_output/out_{tag}_check/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)

def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name+'.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)

def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std,
                             dtype=tf.float32) 
    return input_layer + noise

def Generator(n_samples, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 8*6*6*DIM, noise)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*DIM, 6, 6])

    # Added this layer
    output = lib.ops.deconv2d.Deconv2D('Generator.0', 8*DIM, 4*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1.5', [0,2,3], output)    
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,2,3], output)
    output = tf.nn.relu(output)

    output = gaussian_noise_layer(output, 0.1)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,2,3], output)
    output = tf.nn.relu(output)
    
    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, 1, 5, output)
    
    output = tf.nn.sigmoid(output)

    output = tf.reshape(output, [-1, OUTPUT_DIM])
    return output

def Discriminator(inputs):
    output = tf.reshape(inputs, [-1, 1, 96, 96])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',1,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,2,3], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,2,3], output)
    output = LeakyReLU(output)

    # Added this layer
    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if MODE == 'wgan':
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,2,3], output)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 8*6*6*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 8*6*6*DIM, 1, output)

    return tf.reshape(output, [-1])



print("Setting up models")
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
fake_data = Generator(BATCH_SIZE)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

if MODE == 'wgan':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.RMSPropOptimizer(
        learning_rate=5e-5
    ).minimize(disc_cost, var_list=disc_params)

    clip_ops = []
    for var in lib.params_with_name('Discriminator'):
        clip_bounds = [-.01, .01]
        clip_ops.append(
            tf.assign(
                var,
                tf.clip_by_value(var, clip_bounds[0], clip_bounds[1])
            )
        )
    clip_disc_weights = tf.group(*clip_ops)

elif MODE == 'wgan-gp':
    gen_cost = -tf.reduce_mean(disc_fake)
    disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    alpha = tf.random_uniform(
        shape=[BATCH_SIZE,1],
        minval=0.,
        maxval=1.
    )
    differences = fake_data - real_data
    interpolates = real_data + (alpha*differences)
    gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    disc_cost += LAMBDA*gradient_penalty

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, #1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=1e-4, #1e-4,
        beta1=0.5,
        beta2=0.9
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

elif MODE == 'dcgan':
    gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.ones_like(disc_fake)
    ))

    disc_cost =  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_fake,
        tf.zeros_like(disc_fake)
    ))
    disc_cost += tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        disc_real,
        tf.ones_like(disc_real)
    ))
    disc_cost /= 2.

    gen_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(gen_cost, var_list=gen_params)
    disc_train_op = tf.train.AdamOptimizer(
        learning_rate=2e-4,
        beta1=0.5
    ).minimize(disc_cost, var_list=disc_params)

    clip_disc_weights = None

# For saving samples
fixed_noise = tf.constant(np.random.normal(size=(128, 128)).astype('float32'))
fixed_noise_samples = Generator(128, noise=fixed_noise)
def generate_image(frame, true_dist):
    samples = session.run(fixed_noise_samples)
    lib.save_images.save_images(
        samples.reshape((128, NSIDE, NSIDE)),
        out_dir+'samples_{}.png'.format(frame)
    )

# Dataset iterator
train_data = lib.datautils.load_numpy(imarr_fn)
train_gen = lib.datautils.DataGenerator(train_data, batch_size=BATCH_SIZE)

# To save model
#saver = tf.train.Saver(max_to_keep=4)
#saver = tf.train.Saver()

print("Training")
# Train loop
with tf.Session() as session:

    session.run(tf.initialize_all_variables())

    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _gen_cost, _ = session.run(
                [gen_cost, gen_train_op])

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _ = train_gen.next()
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        if iteration > 0:
            lib.plot.plot(out_dir+'train gen cost', _gen_cost)
        lib.plot.plot(out_dir+'train disc cost', _disc_cost)
        lib.plot.plot(out_dir+'time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if (iteration % SAMPLE_ITERS) == 0 or (iteration==ITERS-1):

            generate_image(iteration, _data)
    

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % SAMPLE_ITERS == 0) \
          or (iteration==ITERS-1):
            lib.plot.flush()

        #if (iteration % SAVE_ITERS == 0):
        #    saver.save(session, out_dir+'model', global_step=iteration)

        lib.plot.tick()
