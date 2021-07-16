# *****************************************************
# * File Name : anomaly_detection_batch.py
# * Creation Date : 2019-08-04
# * Created By : kstoreyf
# * Description : Computes anomaly scores for images
#       using the pre-trained generator, discriminator
#       and encoder.
# *****************************************************

import os, sys
sys.path.append(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import h5py

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.mnist
import tflib.plot
import tflib.datautils

# General arameters
DIM = 64
NSIDE = 96
NBANDS = 3
OUTPUT_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 10
#BATCH_SIZE = 1000 #orig!
ITERS = 10

tag = 'gri_1k'
imarr_fn = f'/scratch/ksf293/anomalies/data/images_h5/images_{tag}.h5'
lambda_weight = 0.3
savetag = f'_lambda{lambda_weight}_checkresid'
startcount = 0

# Generator and discriminator parameters
gentag = 'gri_save'
gennum = 10000
gen_fn = f'/scratch/ksf293/anomalies/training_output/wgan_training/wgan_{gentag}/model-gen-{gennum}'

disctag = gentag
discnum = gennum
disc_fn = f'/scratch/ksf293/anomalies/training_output/wgan_training/wgan_{disctag}/model-disc-{discnum}'

# Encoder parameters (for initial first guess)
enctag = f'gri_lambda{lambda_weight}'
encnum = 5000
enc_fn = f'/scratch/ksf293/anomalies/training_output/encoder_training/encoder_{enctag}/model-encoder-{encnum}'

result_fn = f'/scratch/ksf293/anomalies/results/results_{tag}{savetag}.h5'

# Running detector
print(f"Running anomaly detection for {tag} with generator {gentag}")

print("Loading trained models")
Generator = hub.Module(gen_fn)
Discriminator = hub.Module(disc_fn)
Encoder = hub.Module(enc_fn)
print("Loaded")

print("Setting up model")
real = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])
noise = np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')
z = tf.get_variable(name='ano_z', initializer=noise, validate_shape=False)
z_plhdr = tf.placeholder(tf.float32)
z_ass = z.assign(z_plhdr)
reconstructed = Generator(z)

# Compute residuals, anomaly score
feature_residual = tf.reduce_sum(tf.abs(tf.subtract(
                    Discriminator(real, signature='feature_match'),
                    Discriminator(reconstructed, signature='feature_match')
                    )), axis=[1,2,3]) 
feature_residual = tf.Print(feature_residual, [feature_residual], message="This is feature_residual: ")
residual_image = tf.abs(tf.subtract(real, reconstructed))
residual_image = tf.Print(residual_image, [residual_image], message="This is resid_image: ")
#residual_image_shape = tf.Print(residual_image.get_shape(), [residual_image.get_shape()], message="This is resid_image shape: ")
residual = tf.reduce_sum(residual_image, axis=1)

residual = tf.Print(residual, [residual], message="This is resid: ")
#residual.get_shape()
#tf.Print("hi")#, output_stream=sys.stdout)
#print("uh")
#tf.Print(real.shape)#, output_stream=sys.stdout)
#tf.Print(tf.abs(tf.subtract(real, reconstructed)))
#tf.Print(residual.shape)
anomaly_score = (1-lambda_weight)*residual + lambda_weight*feature_residual

# Optimize
t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'ano_z' in var.name]
optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-1, #1e-4,
        beta1=0.4,
        beta2=0.9
    ).minimize(anomaly_score, var_list=params)

# If result file doesn't yet exist, create new h5 datasets.
# If it does, figure out where to start adding.
if not os.path.isfile(result_fn):
    print(f"Making new result file at {result_fn}")
    fres = h5py.File(result_fn,"w")
    fres.create_dataset('idxs', (0,), maxshape=(None,), chunks=(BATCH_SIZE,))
    fres.create_dataset('object_ids', (0,), maxshape=(None,), chunks=(BATCH_SIZE,))
    fres.create_dataset('reconstructed', (0,NSIDE,NSIDE,NBANDS), maxshape=(None,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    fres.create_dataset('gen_scores', (0,), maxshape=(None,), chunks=(BATCH_SIZE,))
    fres.create_dataset('disc_scores', (0,), maxshape=(None,), chunks=(BATCH_SIZE,))
    fres.create_dataset('anomaly_scores', (0,), maxshape=(None,), chunks=(BATCH_SIZE,))
    count = 0
    fres.attrs['count'] = count
    fres.close()
else:
    fres = h5py.File(result_fn,"r")
    count = fres.attrs['count']
    fres.close()
    print(f"Loaded result file at {result_fn}, count = {count}")

# Set up data generator
print("Loading data")
data = lib.datautils.load(imarr_fn, dataset='images')
print("data")
idxs = lib.datautils.load(imarr_fn, dataset='idxs')
print("idxs")
object_ids = lib.datautils.load(imarr_fn, dataset='object_ids')
print("object_ids")
indices_now = np.arange(data.len())
print("Initializing generator")
data_gen = lib.datautils.DataGenerator(data, y=indices_now, batch_size=BATCH_SIZE, luptonize=True, shuffle=False, starti=count, once=True)
print(f'Num to detect: {len(data)}')

def resize_datasets(f, addsize):
    datasets = ['idxs', 'object_ids', 'reconstructed', 'gen_scores', \
                'disc_scores', 'anomaly_scores']
    for dataset in datasets:
        f[dataset].resize(f[dataset].shape[0]+addsize, axis=0)

# Run detection loop
start = time.time()
moredata = True
nbatches = 0
loc = startcount*BATCH_SIZE
while moredata:
    fres = h5py.File(result_fn,"a")
    
    with tf.Session() as sess:
        s0 = time.time() 
        sess.run(tf.global_variables_initializer())
        print(f'Batch {nbatches}, count {count}')
        nbatches += 1
        
        print('getting images')        
        _images, _indices_now = data_gen.next()
        nimages = len(_images)
        if data_gen.is_done:
            moredata = False
        
        print('resizing datasets')
        resize_datasets(fres, nimages)

        print('encoding')
        _zinit_tensor = Encoder(_images)
        _zinit = sess.run(_zinit_tensor)
        
        # Add padding
        if nimages<BATCH_SIZE:
            _zinit_padded = np.zeros((BATCH_SIZE, 128))
            _zinit_padded[:nimages] = _zinit
            _images_padded = np.zeros((BATCH_SIZE, OUTPUT_DIM))
            _images_padded[:nimages] = _images
            _zinit = _zinit_padded
            _images = _images_padded

        print('running')
        # This was how we got feeding an initial value to work! Cred to @riblidezso
        _z = sess.run([z])        
        _z_ass = sess.run([z_ass], feed_dict={z_plhdr:_zinit})
        
        _z = sess.run([z])
        feed_dict={real: _images}
        for j in range(ITERS):
            _residual, _feature_residual, _score, _reconstructed, _ = sess.run(
                [residual, feature_residual, anomaly_score, reconstructed, optimizer],
                feed_dict=feed_dict)

        _reconstructed = _reconstructed.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
        _reconstructed = (255.*_reconstructed).astype('uint8')
        
        print('saving')
        for bb in range(nimages):
            idx = idxs[_indices_now[bb]]
            objid = object_ids[_indices_now[bb]]
            fres['idxs'][count] = idx
            fres['object_ids'][count] = objid
            fres['reconstructed'][count, ...] = _reconstructed[bb]
            fres['gen_scores'][count] = _residual[bb]
            fres['disc_scores'][count] = _feature_residual[bb]
            fres['anomaly_scores'][count] = _score[bb]
            fres.attrs['count'] = count+1
            count += 1
        
        e0 = time.time()
        print(f't iter: {e0-s0}')
                        
    fres.close() 

end = time.time()
print(f"Time for {len(data)} images: {end-start} s")

print("Done")
