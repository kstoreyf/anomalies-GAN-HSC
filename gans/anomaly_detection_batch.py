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
NSIDE = 96
OUTPUT_DIM = NSIDE*NSIDE
BATCH_SIZE = 1000
#BATCH_SIZE = 1
ITERS = 10

#tag = 'i20.0_norm_100k'
tag = 'i20.0_norm'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
savetag = '_features0.05go'
startcount = 942

gentag = 'i20.0_norm_features'
gennum = 12000
gen_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/training_output/out_{gentag}/model-gen-{gennum}'

disctag = gentag
discnum = gennum
disc_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/training_output/out_{disctag}/model-disc-{discnum}'

enctag = f'{tag}_features0.05'
encnum = 9000
enc_fn = f'/scratch/ksf293/kavli/anomaly/training_output/encoder_{enctag}/model-encoder-{encnum}'

results_dir = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}{savetag}'
if not os.path.isdir(results_dir):
    os.mkdir(results_dir)
#result_fn = f'{results_dir}/results_{tag}{savetag}.npy'

print(f"Running anomaly detection for {tag} with generator {gentag}")

print("Loading trained models")
Generator = hub.Module(gen_fn)
Discriminator = hub.Module(disc_fn)
Encoder = hub.Module(enc_fn)
print("Loaded")
print("Setting up model")

real = tf.placeholder(tf.float32, shape=[None, OUTPUT_DIM])

noise = 100*np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')
z = tf.get_variable(name='ano_z', initializer=noise, validate_shape=False)

z_plhdr = tf.placeholder(tf.float32)
z_ass = z.assign(z_plhdr)

reconstructed = Generator(z)

feature_residual = tf.reduce_sum(tf.abs(tf.subtract(
                    Discriminator(real, signature='feature_match'),
                    Discriminator(reconstructed, signature='feature_match')
                    )), axis=[1,2,3]) 

residual = tf.reduce_sum(tf.abs(tf.subtract(real, reconstructed)), axis=1)

anomaly_weight = 0.05
anomaly_score = (1-anomaly_weight)*residual + anomaly_weight*feature_residual

t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'ano_z' in var.name]

optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-1, #1e-4,
        beta1=0.4,
        beta2=0.9
    ).minimize(anomaly_score, var_list=params)

print("Loading data")
data = lib.datautils.load_numpy(imarr_fn)
idxs = range(len(data))

#idxs = [8227, 23384, 27990, 51660, 53654, 75000, 77166, 79478, 136646, 154749, 168076, 169037, 205029, 222372, 230340, 233239, 306282, 371503, 391733, 393979, 403224, 430453, 458481, 537727, 544052, 595328, 646272, 696868, 716897, 786702, 826764, 837307]
#data = data[idxs]
print(f'Num to detect: {len(data)}')


#with tf.Session() as sess:
#sess.run(tf.global_variables_initializer())
    
start = time.time()
    
moredata = 1
count = startcount
loc = startcount*BATCH_SIZE
while moredata:
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'Batch {count}')
        _images = data[loc:loc+BATCH_SIZE].reshape((-1, OUTPUT_DIM))
        idx = idxs[loc:loc+BATCH_SIZE]
        
        #if (len(idx))<BATCH_SIZE:
            
        
        _zinit_tensor = Encoder(_images)
        _zinit = sess.run(_zinit_tensor)
                
        s0 = time.time() 

        # This was how we got feeding an initial value to work! Cred to @riblidezso
        _z = sess.run([z])        
        _z_ass = sess.run([z_ass], feed_dict={z_plhdr:_zinit})
        
        _z = sess.run([z])
        #_zinit = np.random.normal(size=(BATCH_SIZE, 128)).astype('float32')
        feed_dict={real: _images}
        for j in range(ITERS):
            _residual, _feature_residual, _score, _reconstructed, _ = sess.run(
                [residual, feature_residual, anomaly_score, reconstructed, optimizer],
                feed_dict=feed_dict)
        e0 = time.time()
        print(f't iter: {e0-s0}')

        _reconstructed = _reconstructed.reshape((-1,96,96))
        result = []
        #for bb in range(BATCH_SIZE):
        for bb in range(len(idx)):
            result.append([_images[bb], _reconstructed[bb], _residual[bb], _feature_residual[bb], _score[bb], idx[bb]])
        
        loc += BATCH_SIZE
        if loc>=len(data):
            moredata = 0
      
        #if os.path.isfile(result_fn):
        #    os.rename(result_fn, f'{result_fn[:-4]}-backup.png')
        #np.save(result_fn, np.array(result))
        np.save( f'{results_dir}/results_{tag}{savetag}-{count}.npy', np.array(result))
        count += 1
    end = time.time()
    print(f"Time for {len(data)} images: {end-start} s")

print("Done")
