# ******************************************************
# * File Name : autoencoder_rgb.py
# * Creation Date : 2019-08-07
# * Created By : kstoreyf
# * Description : Trains an autoencoder on the residuals
#       of images to find a latent-space representation,
#       to be used as input for UMAP clustering
# ******************************************************
import os
import sys
import shutil
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time

import numpy as np
import tensorflow as tf
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
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
DIM = 64 #dimension of autoencoder convolutions
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 30
ITERS = 30500 # How many generator iterations to train for
SAMPLE_ITERS = 500 # Multiples at which to generate image sample
SAVE_ITERS = 500 # Multiples at which to save the autoencoder state
overwrite = True
LATENT_DIM = 64


imtag = 'gri_lambda0.3_3sigd'
tag = 'gri_lambda0.3_3sigd'
base_dir = '/scratch/ksf293/anomalies'
results_fn = f'{base_dir}/results/results_{tag}.h5'
imarr_fn = f'{base_dir}/data/images_h5/images_{imtag}.h5'
mode = "reals" # one of: ['reals', 'residuals', 'disc_features_real']

if 'disc' in mode:
    NSIDE = 6
    NBANDS = 512
    IMAGE_DIM = NSIDE*NSIDE*NBANDS
    save_ims = False
else:
    save_ims = True

savetag = f'_latent{LATENT_DIM}_{mode}_long'

out_dir = f'{base_dir}/training_output/autoencoder_training/autoencoder_{tag}{savetag}/'
loss_fn = f'{out_dir}loss.txt'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


def AutoEncoder_module():
    # Inputs are 3D images
    inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
    output = tf.reshape(inputs, [-1, NBANDS, NSIDE, NSIDE])
    
    # Compression
    # 96x96
    output = lib.ops.conv2d.Conv2D('AutoEncoder.1', NBANDS, DIM,5,output,stride=2)
    output = tf.nn.relu(output)
    #48x48
    output = lib.ops.conv2d.Conv2D('AutoEncoder.2', DIM, 2*DIM, 5, output, stride=2)
    output = tf.nn.relu(output)
    #24x24
    output = lib.ops.conv2d.Conv2D('AutoEncoder.3', 2*DIM, 4*DIM, 5, output, stride=2)
    output = tf.nn.relu(output)
    #12x12
    output = lib.ops.conv2d.Conv2D('AutoEncoder.4', 4*DIM, 8*DIM, 5, output, stride=2)
    output = tf.nn.relu(output)
    #6x6

    output = tf.layers.flatten(output)
    output_latent = tf.layers.dense(output, LATENT_DIM, activation=None)
    
    # Outputs are latent-space representations of images
    hub.add_signature(inputs=inputs, outputs=output_latent, name='latent')

    # Decompressioin
    activation = None
    output = tf.layers.dense(output_latent, DIM*6*6, use_bias=False, activation=activation)
    output = tf.reshape(output, [-1, DIM, 6, 6])
    
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.6', DIM, 4*DIM, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.7', 4*DIM, 2*DIM, 5, output)
    output = tf.nn.relu(output)
   
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.8', 2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.9', DIM, NBANDS, 5, output)
    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, IMAGE_DIM])

    # inputs are latent representation, outputs are reconstructed images
    hub.add_signature(inputs=output_latent, outputs=output, name='decode_latent')
    hub.add_signature(inputs=inputs, outputs=output)

    return output

# Run autoencoder
autoencoder_spec = hub.create_module_spec(AutoEncoder_module)
AutoEncoder = hub.Module(autoencoder_spec, name='AutoEncoder', trainable=True)

residual_orig = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
residual_reconstructed = AutoEncoder(residual_orig)
loss = tf.reduce_mean(tf.square(residual_reconstructed - residual_orig))

t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'AutoEncoder' in var.name]

ae_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-3
    ).minimize(loss, var_list=params)

# Load data
print("Loading data")
data = lib.datautils.load(results_fn, dataset=mode)

data_gen = lib.datautils.DataGenerator(data, batch_size=BATCH_SIZE, luptonize=False, normalize=False, smooth=False)
fixed_im, _ = data_gen.sample(128)

n=10
#fixed latent space rep to test reencoding
fixed_im_samples = AutoEncoder(fixed_im.reshape((-1,IMAGE_DIM))) 
print("Fixed im samples") # dim (128, 27648)
print(fixed_im_samples.shape)
fixed_im = fixed_im.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
if save_ims:
    lib.save_images.save_images(
        fixed_im,
        out_dir+'real.png',
        unnormalize=False
         )
def generate_image(frame):
    samples = sess.run(fixed_im_samples)
    samples = samples.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
    lib.save_images.save_images(
        samples,
        out_dir+'samples_{}.png'.format(frame),
        unnormalize=False
    )

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for iteration in range(ITERS):
        start_time = time.time()
        _data, _ = data_gen.next()

        _loss, _ = sess.run(
            [loss, ae_optimizer],
            feed_dict={residual_orig: _data}
        )
        if iteration >= 100:
            lib.plot.plot(out_dir+'autoencoder loss', _loss)

        if (iteration < 5):
            lib.plot.flush()
        if (iteration % SAMPLE_ITERS == 0) or (iteration==ITERS-1):
            lib.plot.flush()
        if (iteration % SAVE_ITERS == 0) and iteration>0:
            print(iteration)
            if save_ims:
                generate_image(iteration)
            ae_fn = out_dir+f'model-autoencoder-{iteration}'
            if overwrite and os.path.isdir(ae_fn):
                shutil.rmtree(ae_fn)
            AutoEncoder.export(ae_fn, sess)
        lib.plot.tick()

