# ******************************************************
# * File Name : autoencoder.py
# * Creation Date : 2019-08-07
# * Created By : kstoreyf
# * Description : Trains an autoencoder on the residuals
#       of images to find a latent-space represenation,
#       to be used as input to a clustering algorithm
# ******************************************************
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

import utils


DIM = 64 #dimension of autoencoder convolutions
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 32
ITERS = 30000#10000 # How many generator iterations to train for
SAMPLE_ITERS = 1000 # Multiples at which to generate image sample
SAVE_ITERS = 1000 # Multiples at which to save the autoencoder state
overwrite = True
LATENT_DIM = 64

#tag = 'gri'
#tag = 'gri_3sig'
#tag = 'gri_cosmos'
tag = 'gri_100k'
results_dir = '/scratch/ksf293/kavli/anomaly/results' #may need to move stuff back to scratch from archive
#results_dir = '/archive/k/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.h5'
#results_fn = f'{results_dir}/results_{tag}.npy'
#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5' # NOTE NEW FORMAT
#savetag = '_latent32_lr1e-2'
#savetag = f'_latent{LATENT_DIM}_real'
savetag = f'_latent{LATENT_DIM}'
#savetag = '_aereal'

out_dir = f'/scratch/ksf293/kavli/anomaly/training_output/autoencoder_{tag}{savetag}/'
loss_fn = f'{out_dir}/loss.txt'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

#lib.print_model_settings(locals().copy())

def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

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
    #output = lib.ops.conv2d.Conv2D('AutoEncoder.5', 8*DIM, int(DIM/64), 5, output)
    #output = tf.nn.relu(output)
    
    #output = tf.reshape(tf.layers.flatten(output), (-1, LATENT_DIM))
    output = tf.layers.flatten(output)
    output_latent = tf.layers.dense(output, LATENT_DIM, activation=None)
    
    #tf.Print(output_latent)
    #print(output_latent)
    # Outputs are latent-space representations of images
    hub.add_signature(inputs=inputs, outputs=output_latent, name='latent')

    # Decompressioin
    #output = tf.reshape(
    activation = None
    output = tf.layers.dense(output_latent, DIM*6*6, use_bias=False, activation=activation) #should this be DIM*6*6*8? bc 8*DIM in last conv layer; see discriminator / generator
    output = tf.reshape(output, [-1, DIM, 6, 6]) #not sure if need to change for RGB
    
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


autoencoder_spec = hub.create_module_spec(AutoEncoder_module)
AutoEncoder = hub.Module(autoencoder_spec, name='AutoEncoder', trainable=True)

residual_orig = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
residual_reconstructed = AutoEncoder(residual_orig)
loss = tf.reduce_mean(tf.square(residual_reconstructed - residual_orig))

t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'AutoEncoder' in var.name]

ae_optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-3 #1e-4,
        #beta1=0.4,
        #beta2=0.9
    ).minimize(loss, var_list=params)

print("Loading data")
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(
                                                    results_fn, imarr_fn)
residuals, reals, recons = utils.get_residuals(reals, recons)

data = residuals
#print("AUTOENCODING REALS (NOT RESIDUALS)")
#data = reals
data_gen = lib.datautils.DataGenerator(data, batch_size=BATCH_SIZE, luptonize=False, normalize=False, smooth=False)
fixed_im, _ = data_gen.sample(128)

n=10
print("fixed")
print(fixed_im[0][:n])
print(fixed_im.reshape((-1,IMAGE_DIM))[0][:n])
fixed_im_samples = AutoEncoder(fixed_im.reshape((-1,IMAGE_DIM))) #fixed latent space rep to test reencoding
print("Fixed im samples") # dim (128, 27648)
print(fixed_im_samples.shape)
fixed_im = fixed_im.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
lib.save_images.save_images(
        fixed_im,
        out_dir+'real.png',
        unnormalize=False
    )
def generate_image(frame):
    samples = sess.run(fixed_im_samples)
    #print("samples")
    #print(samples[0][:n])
    samples = samples.reshape((128, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
    #print(samples[0][:n])
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
            #generate_image(iteration)
        if (iteration % SAVE_ITERS == 0) and iteration>0:
            generate_image(iteration)
            ae_fn = out_dir+f'model-autoencoder-{iteration}'
            if overwrite and os.path.isdir(ae_fn):
                shutil.rmtree(ae_fn)
            AutoEncoder.export(ae_fn, sess)
        lib.plot.tick()

