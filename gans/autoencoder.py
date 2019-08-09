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


DIM = 64
#DIM = 2
NSIDE = 96
IMAGE_DIM = NSIDE*NSIDE
BATCH_SIZE = 32
ITERS = 10000 # How many generator iterations to train for
SAMPLE_ITERS = 100 # Multiples at which to generate image sample
SAVE_ITERS = 1000
overwrite = True
LATENT_DIM = 64

tag = 'i20.0_norm_100k_features0.05go'
results_dir = '/scratch/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.npy'
#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
savetag = '_latent32_lr1e-2'

out_dir = f'/scratch/ksf293/kavli/anomaly/training_output/autoencoder_{tag}{savetag}/'
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

lib.print_model_settings(locals().copy())


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha*x, x)

def AutoEncoder_module():
    inputs = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
    output = tf.reshape(inputs, [-1, 1, 96, 96])
    
    # Compression
    # 96x96
    output = lib.ops.conv2d.Conv2D('AutoEncoder.1', 1, DIM,5,output,stride=2)
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
    print(output_latent)
    hub.add_signature(inputs=inputs, outputs=output_latent, name='latent')

    # Decompressioin
    #output = tf.reshape(
    activation = None
    output = tf.layers.dense(output_latent, DIM*6*6, use_bias=False, activation=activation)
    output = tf.reshape(output, [-1, DIM, 6, 6])
    
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.6', DIM, 4*DIM, 5, output)
    output = tf.nn.relu(output)

    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.7', 4*DIM, 2*DIM, 5, output)
    output = tf.nn.relu(output)
   
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.8', 2*DIM, DIM, 5, output)
    output = tf.nn.relu(output)
    
    output = lib.ops.deconv2d.Deconv2D('AutoEncoder.9', DIM, 1, 5, output)
    output = tf.nn.relu(output)

    output = tf.reshape(output, [-1, IMAGE_DIM])

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
#res = lib.datautils.load_numpy(results_fn)
res = np.load(results_fn, allow_pickle=True)
images = res[:,0]
images = np.array([np.array(im) for im in images])
images = images.reshape((-1, NSIDE, NSIDE))
recons = res[:,1]
recons = np.array([np.array(im) for im in recons])
recons = recons.reshape((-1, NSIDE, NSIDE))
data = abs(images-recons)
print(data.shape)

idxs = range(len(data))
data_gen = lib.datautils.DataGenerator(data, batch_size=BATCH_SIZE)

fixed_im = data[:128]
fixed_im_samples = AutoEncoder(fixed_im.reshape((-1,IMAGE_DIM)))
lib.save_images.save_images(
        fixed_im.reshape((128, NSIDE, NSIDE)),
        out_dir+'real.png'
    )
def generate_image(frame):
    samples = sess.run(fixed_im_samples)
    lib.save_images.save_images(
        samples.reshape((128, NSIDE, NSIDE)),
        out_dir+'samples_{}.png'.format(frame)
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

