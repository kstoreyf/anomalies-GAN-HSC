# **************************************************
# * File Name : residuals.py
# * Creation Date : 2019-07-26
# * Last Modified :
# * Created By : 
# * Description :
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


tag = 'i20.0_norm'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
savetag = '_try3'

gentag = 'i20.0_norm_hub'
gennum = 7000
gen_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/training_output/out_{gentag}/model-gen-{gennum}'

results_fn = f'/home/ksf293/kavli/anomalies-GAN-HSC/results/results_{tag}{savetag}.npy'

DIM = 64
NSIDE = 96
OUTPUT_DIM = NSIDE*NSIDE

print(f"Running anomaly detection for {tag} with generator {gentag}")

print("Loading generator")
Generator = hub.Module(gen_fn)
print("Loaded")

def Detector(z):
    output = lib.ops.linear.Linear('Detector.0', 128, DIM, z)
    output = tf.nn.relu(output)
    output = lib.ops.linear.Linear('Detector.1', DIM, 128, output)
    return output

print("Setting up model")
real = tf.placeholder(tf.float32, shape=[1, OUTPUT_DIM])

noise = tf.placeholder(tf.float32, shape=[1, 128])

#ano_z = tf.get_variable('ano_z', shape=[1, 128], dtype = tf.float32, 
#        initializer = tf.random_normal_initializer())

#reconstructed = Generator(ano_z)
reconstructed = Generator(Detector(noise)) # 1 sample
residual = tf.reduce_sum(tf.abs(tf.subtract(real, reconstructed)))

t_vars = tf.trainable_variables()
#params = [var for var in t_vars if 'ano_z' in var.name]
params = [var for var in t_vars if 'Detector' in var.name]

optimizer = tf.train.AdamOptimizer(
        learning_rate=1e-3, #1e-4,
        beta1=0.5#,
        #beta2=0.9
    ).minimize(residual, var_list=params)

data = lib.datautils.load_numpy(imarr_fn)
data_gen = lib.datautils.DataGenerator(data)

results = []
with tf.Session() as sess:
    
    sess.run(tf.global_variables_initializer())

    for i in range(len(data)): 

        image, _ = data_gen.next_one()
        _noise = np.random.normal(size=(1, 128)).astype('float32')
        
        for j in range(200):
            #_residual, _reconstructed, _ano_z, _ = sess.run([residual, reconstructed, ano_z, optimizer],
            _residual, _reconstructed, _ = sess.run([residual, reconstructed, optimizer],
                feed_dict={real: image.reshape(1, OUTPUT_DIM)#})
                                    , noise: _noise})
        results.append([image, _reconstructed, _residual])
        #print(_ano_z[0][:10])
        if (i % 100 == 0) and i>0:
            print(i, _residual)
            np.save(results_fn, np.array(results))
    

