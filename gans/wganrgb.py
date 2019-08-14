# *********************************************************************************
# * File Name : wganrgb.py
# * Creation Date : 2019-08-12
# * Created By : kstoreyf
# * Description : Implemtation of a WGAN-GP to generate 
# *     images from 96x96 galaxy cutouts, in 3 bands. Based on
# *     https://github.com/igul222/improved_wgan_training/blob/master/gan_mnist.py.
# *     Only tested on wgan-gp mode. Uses tensorflow-hub for saving models.
# *********************************************************************************

import os, sys
sys.path.append(os.getcwd())

import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot
import tflib.datautils

#np.set_printoptions(threshold=sys.maxsize)

MODE = 'wgan-gp' # dcgan, wgan, or wgan-gp
DIM = 64 # Model dimensionality
BATCH_SIZE = 32 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 50000 # How many generator iterations to train for
SAMPLE_ITERS = 100 # Multiples at which to generate image sample
SAVE_ITERS = 500
NSIDE = 96 # Don't change this without changing the model layers!
NBANDS = 3
OUTPUT_DIM = NSIDE*NSIDE*NBANDS # Number of pixels in MNIST (28*28)
batchnorm = False

tag = 'gri'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
#tag = 'i20.0_norm'
#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'

out_dir = f'/scratch/ksf293/kavli/anomaly/training_output/out_{tag}_save/'
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

def Generator_module():
    noise = tf.placeholder(tf.float32, shape=[None, 128])

    output = lib.ops.linear.Linear('Generator.Input', 128, 8*6*6*DIM, noise)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1', [0], output)
    output = tf.nn.relu(output)
    output = tf.reshape(output, [-1, 8*DIM, 6, 6])

    output = lib.ops.deconv2d.Deconv2D('Generator.0', 8*DIM, 4*DIM, 5, output)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN1.5', [0,3,2], output)    
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('Generator.2', 4*DIM, 2*DIM, 5, output)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN2', [0,3,2], output)
    output = tf.nn.relu(output)

    output = gaussian_noise_layer(output, 0.1)

    output = lib.ops.deconv2d.Deconv2D('Generator.3', 2*DIM, DIM, 5, output)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Generator.BN3', [0,3,2], output)
    output = tf.nn.relu(output)
    
    output = lib.ops.deconv2d.Deconv2D('Generator.5', DIM, NBANDS, 5, output)
    
    output = tf.nn.sigmoid(output)
    #output = tf.tanh(output)

    output = tf.reshape(output, [-1, OUTPUT_DIM])
    
    hub.add_signature(inputs=noise, outputs=output)
    return output

def Discriminator_module():
    inputs = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
    output = tf.reshape(inputs, [-1, NBANDS, NSIDE, NSIDE])

    output = lib.ops.conv2d.Conv2D('Discriminator.1',NBANDS,DIM,5,output,stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', DIM, 2*DIM, 5, output, stride=2)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN2', [0,3,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 2*DIM, 4*DIM, 5, output, stride=2)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN3', [0,3,2], output)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.4', 4*DIM, 8*DIM, 5, output, stride=2)
    if batchnorm:
        output = lib.ops.batchnorm.Batchnorm('Discriminator.BN4', [0,3,2], output)
    output = LeakyReLU(output)
    
    hub.add_signature(inputs=inputs, outputs=output, name='feature_match')

    output = tf.reshape(output, [-1, 8*6*6*DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 8*6*6*DIM, 1, output)

    tf.reshape(output, [-1])

    hub.add_signature(inputs=inputs, outputs=output)
    return output


print("Setting up models")

real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
generator_spec = hub.create_module_spec(Generator_module)
Generator = hub.Module(generator_spec, name='Generator', trainable=True)

discriminator_spec = hub.create_module_spec(Discriminator_module)
Discriminator = hub.Module(discriminator_spec, name='Discriminator', trainable=True)

noise = tf.placeholder(tf.float32, shape=[BATCH_SIZE, 128])
fake_data = Generator(noise)

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

gen_params = [v for v in tf.trainable_variables() if 'Generator' in v.name]
disc_params = [v for v in tf.trainable_variables() if 'Discriminator' in v.name]

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
fixed_noise_samples = Generator(fixed_noise)
def generate_image(frame):
    samples = session.run(fixed_noise_samples)
    #print(samples[0][0])
    #samples = (0.5*(samples+1.))
    #print(samples[0][0])
    samples = samples.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1) #0321 also works
    #samples = samples.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,1,3,2)
    #print(samples[0])
    lib.save_images.save_images(
        samples, out_dir+'samples_{}.png'.format(frame), unnormalize=True
    )

# Dataset iterator
print("Loading data")
train_data = lib.datautils.load(imarr_fn)
train_gen = lib.datautils.DataGenerator(train_data, batch_size=BATCH_SIZE, luptonize=True, normalize=False, smooth=False)

print("Writing real sample")
sample_real, _ = train_gen.sample(128)
sample_real = sample_real.reshape((128, NBANDS, NSIDE, NSIDE))
sample_real = sample_real.transpose(0,2,3,1)
#sample_real = sample_real.reshape((128, NSIDE, NSIDE, NBANDS))
lib.save_images.save_images(sample_real, out_dir+'real.png', unnormalize=True)
#print(sample_real[0])

print("Training")
# Train loop
with tf.Session() as session:

    session.run(tf.global_variables_initializer())

    for iteration in range(ITERS):
        start_time = time.time()

        if iteration > 0:
            _noise = np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')
            _gen_cost, _ = session.run(
                [gen_cost, gen_train_op],
                feed_dict={noise: _noise})

        if MODE == 'dcgan':
            disc_iters = 1
        else:
            disc_iters = CRITIC_ITERS
        for i in range(disc_iters):
            _data, _ = train_gen.next()
            #print(_data.shape)
            #print((_data[0][int(OUTPUT_DIM/2-24):int(OUTPUT_DIM/2+24)]*255.).astype('uint8'))
            #print(_data[0][40:-40,40:-40])
            _noise = np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')
            
            _disc_cost, _ = session.run(
                [disc_cost, disc_train_op],
                feed_dict={real_data: _data,
                           noise: _noise}
            )
            if clip_disc_weights is not None:
                _ = session.run(clip_disc_weights)

        if iteration > 0:
            lib.plot.plot(out_dir+'train gen cost', _gen_cost)
        lib.plot.plot(out_dir+'train disc cost', _disc_cost)
        lib.plot.plot(out_dir+'time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        if (iteration % SAMPLE_ITERS) == 0 or (iteration==ITERS-1):
            generate_image(iteration)

        # Write logs every 100 iters
        if (iteration < 5) or (iteration % SAMPLE_ITERS == 0) \
          or (iteration==ITERS-1):
            lib.plot.flush()

        if (iteration % SAVE_ITERS == 0) and iteration>0:
            Generator.export(out_dir+f'model-gen-{iteration}', session)
            Discriminator.export(out_dir+f'model-disc-{iteration}', session)
        lib.plot.tick()
