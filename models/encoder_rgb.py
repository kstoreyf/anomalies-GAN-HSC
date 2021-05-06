# **************************************************
# * File Name : encoder_rgb.py
# * Creation Date : 2019-08-03
# * Created By : kstoreyf
# * Description : Trains an encoder to predict the
#       latent-space representation of an image that
#       minimizes its anomaly score.
# **************************************************
import os
import sys
import shutil
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]="0"
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

# Parameters
DIM = 64
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 32
ITERS = 10000 # How many generator iterations to train for
SAMPLE_ITERS = 250 # Multiples at which to generate image sample
SAVE_ITERS = 1000
overwrite = True

tag = 'gri'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
lambda_weight = 0.5
savetag = '_lambda{lambda_weight}'

out_dir = f'/scratch/ksf293/kavli/anomaly/training_output/encoder_{tag}{savetag}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

# Disc and gen params
gentag = 'gri_save'
gennum = 10000
gen_fn = f'/scratch/ksf293/kavli/anomaly/training_output/wgan_{gentag}/model-gen-{gennum}'

disctag = gentag
discnum = gennum
disc_fn = f'/scratch/ksf293/kavli/anomaly/training_output/wgan_{disctag}/model-disc-{discnum}'

lib.print_model_settings(locals().copy())

print("Loading generator and discriminator")
Generator = hub.Module(gen_fn)
Discriminator = hub.Module(disc_fn)
print("Loaded")

def Encoder_module():
    inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
    output = tf.reshape(inputs, [-1, NBANDS, NSIDE, NSIDE])

    output = lib.ops.conv2d.Conv2D('Encoder.1', NBANDS, DIM,5,output,stride=2)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D('Encoder.2', DIM, 2*DIM, 5, output, stride=2)
    output = tf.nn.relu(output)

    output = lib.ops.conv2d.Conv2D('Encoder.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = tf.nn.relu(output)
    
    output = lib.ops.conv2d.Conv2D('Encoder.4', 4*DIM, 8*DIM, 5, output, stride=2)    
    output = tf.nn.relu(output)
    
    output = tf.reshape(output, [-1, 8*6*6*DIM])

    output = lib.ops.linear.Linear('Encoder.Output', 8*6*6*DIM, 128, output)
    output = tf.tanh(1e-7*output)

    hub.add_signature(inputs=inputs, outputs=output)

    return output

# Set up encoder, compute residuals
real = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
encoder_spec = hub.create_module_spec(Encoder_module)
Encoder = hub.Module(encoder_spec, name='Encoder', trainable=True)

z = Encoder(real)
reconstructed = Generator(z)

feature_residual = tf.reduce_sum(tf.abs(tf.subtract(
                        Discriminator(real, signature='feature_match'),
                        Discriminator(reconstructed, signature='feature_match')
                    )), axis=[1,2,3])
residual = tf.reduce_sum(tf.abs(tf.subtract(real, reconstructed)), axis=1)

anomaly_score = (1-lambda_weight)*residual + lambda_weight*feature_residual

enc_cost = tf.reduce_sum(anomaly_score)

# Optimize
t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'Encoder' in var.name]
enc_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-2, #1e-4,
        beta1=0.4,
        beta2=0.9
    ).minimize(enc_cost, var_list=params)

data = lib.datautils.load(imarr_fn)
idxs = range(len(data))
data_gen = lib.datautils.DataGenerator(data, batch_size=BATCH_SIZE, luptonize=True, normalize=False, smooth=False)

fixed_im, _ = data_gen.sample(128)
fixed_im_samples = Generator(Encoder(fixed_im.reshape((-1,IMAGE_DIM))))
fixed_im = fixed_im.reshape((128, NBANDS, NSIDE, NSIDE))
fixed_im = fixed_im.transpose(0,2,3,1)
lib.save_images.save_images(
        fixed_im,
        out_dir+'real.png', unnormalize=True
    )
def generate_image(frame):
    samples = sess.run(fixed_im_samples)
    lib.save_images.save_images(
        samples.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1),
        out_dir+'samples_{}.png'.format(frame)
    )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(ITERS):
        start_time = time.time()
        _data, _ = data_gen.next()

        _enc_cost, _z, _ = sess.run(
            [enc_cost, z, enc_optimizer],
            feed_dict={real: _data}
        )
        lib.plot.plot(out_dir+'encoder cost', _enc_cost)

        if (iteration < 5):
            lib.plot.flush()
        if (iteration % SAMPLE_ITERS == 0) or (iteration==ITERS-1):
            print("iteration {}, generating samples".format(iteration))
            lib.plot.flush()
            generate_image(iteration)
        if (iteration % SAVE_ITERS == 0) and iteration>0:
            enc_fn = out_dir+f'model-encoder-{iteration}'
            if overwrite and os.path.isdir(enc_fn):
                shutil.rmtree(enc_fn)
            Encoder.export(enc_fn, sess)
        lib.plot.tick()

