# **************************************************
# * File Name : anomaly_detection.py
# * Creation Date : 2019-07-26
# * Created By : kstoreyf
# * Description : Trains a mini-NN to find closest 
#       GAN representation of the image, and assigns
#       anomaly scores based on this.
# **************************************************

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



#tag = 'i20.0_norm_100k'
tag = 'i1k_96x96_norm'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/imarrs_np/hsc_{tag}.npy'
#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
savetag = '_cpu'
mode = 'detector'
#mode = 'simple'

gentag = 'i20.0_norm_hub'
gennum = 7000
gen_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/training_output/out_{gentag}/model-gen-{gennum}'

disctag = 'i20.0_norm_feat'
discnum = 600
disc_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/training_output/out_{disctag}/model-disc-{discnum}'

results_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/results/results_{tag}{savetag}.npy'

DIM = 64
NSIDE = 96
OUTPUT_DIM = NSIDE*NSIDE




print("Loading data")
data = lib.datautils.load_numpy(imarr_fn)
data_gen = lib.datautils.DataGenerator(data)
print(f"Running anomaly detection for {tag} with generator {gentag}")

print("Loading generator and discriminator")
Generator = hub.Module(gen_fn)
Discriminator = hub.Module(disc_fn)
print("Loaded")

def Detector(z):
    #output = lib.ops.linear.Linear('Detector.0', 128, DIM, z)
    output = lib.ops.linear.Linear('Detector.0', 128, 128, z)
    output = tf.nn.relu(output)
    #output = lib.ops.linear.Linear('Detector.1', DIM, 128, output)
    return output

print("Setting up model")
real = tf.placeholder(tf.float32, shape=[1, OUTPUT_DIM])

if mode=='simple':
    noise = tf.get_variable('Detector', shape=[1, 128], dtype = tf.float32, 
            initializer = tf.random_normal_initializer())
    reconstructed = Generator(noise)

if mode=='detector':
    noise = tf.placeholder(tf.float32, shape=[1, 128])
    reconstructed = Generator(Detector(noise)) # 1 sample

feature_residual = tf.reduce_sum(tf.abs(tf.subtract(
                    Discriminator(real, signature='feature_match'),
                    Discriminator(reconstructed, signature='feature_match')
                    ))) 

residual = tf.reduce_sum(tf.abs(tf.subtract(real, reconstructed)))

anomaly_weight = 0.1
anomaly_score = (1-anomaly_weight)*residual + anomaly_weight*feature_residual

t_vars = tf.trainable_variables()
params = [var for var in t_vars if 'Detector' in var.name]

optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-3, #1e-4,
        beta1=0.4,
        beta2=0.9
    ).minimize(anomaly_score, var_list=params)


data_idxs = np.random.choice(len(data), size=32)
#data_idxs = [8227, 23384, 27990, 51660, 53654, 75000, 77166, 79478, 136646, 154749, 168076, 169037, 205029, 222372, 230340, 233239, 306282, 371503, 391733, 393979, 403224, 430453, 458481, 537727, 544052, 595328, 646272, 696868, 716897, 786702, 826764, 837307]
data = data[data_idxs]
print(f'Num to detect: {len(data)}')
results = []
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
config=tf.ConfigProto(gpu_options=gpu_options)

with tf.Session(config=config) as sess:
#with tf.Session() as sess:         
    sess.run(tf.global_variables_initializer())
    start = time.time()
    for i in range(len(data)):
        s0 = time.time()
        #_image, _ = data_gen.next_one()
        _image = data[i]
        _noise = np.random.normal(size=(1, 128)).astype('float32')
        print(i) 
        for j in range(50):
            if mode=='simple':
                feed_dict = {real: _image.reshape(1, OUTPUT_DIM)}
            if mode=='detector':
                feed_dict={real: _image.reshape(1, OUTPUT_DIM), noise: _noise}
            _residual, _feature_residual, _score, _reconstructed, _ = sess.run([residual, feature_residual, anomaly_score, reconstructed, optimizer],
                                                    feed_dict=feed_dict)

        e0 = time.time()
        print(f'time iter: {e0-s0}')
        results.append([_image, _reconstructed, _residual, _feature_residual, _score, data_idxs[i]])
        if (i==len(data)-1) and i>0:
            print(i, _residual)
            np.save(results_fn, np.array(results))
            end = time.time()
            print(f"Time: {end-start} s")

