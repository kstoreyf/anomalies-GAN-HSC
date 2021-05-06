# *****************************************************
# * File Name : discrimator_features_rgb.py
# * Creation Date : 2021-01-11
# * Created By : kstoreyf
# * Description : Saves the discriminator features 
#       (penultimate layer) for the real and recon-
#       structed images, and their residual.
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


NSIDE = 96
NBANDS = 3
NDIM = 512
NFEAT = 6
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 1000

tag = 'gri_lambda0.3_3sigdisc'
#tag = 'gri_10k_lambda0.3'
startcount = 0

disctag = 'gri_save'
discnum = 10000
disc_fn = f'/scratch/ksf293/kavli/anomaly/training_output/wgan_{disctag}/model-disc-{discnum}'

result_fn = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}.h5'

print(f"Running discrimatinator feature saving for {tag}")

print("Loading trained models")
Discriminator = hub.Module(disc_fn)
print("Loaded")
print("Setting up model")

real = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])
reconstructed = tf.placeholder(tf.float32, shape=[None, IMAGE_DIM])

disc_real = Discriminator(real, signature='feature_match')
print(disc_real)
disc_recon = Discriminator(reconstructed, signature='feature_match')
disc_resid = tf.abs(tf.subtract(disc_real, disc_recon))
print(disc_resid)

print(f"Making new datasets for result file at {result_fn}")
#with h5py.File(result_fn,"a") as fres:
fres = h5py.File(result_fn,"a")
if True:
    new_datasets = ['disc_features_real', 'disc_features_recon', 'disc_features_resid']
    for dkey in new_datasets:
        if dkey in fres.keys():
            del fres[dkey]
    print(ksfsdjf)
    fres.create_dataset('disc_features_real', (0,NFEAT,NFEAT,NDIM), maxshape=(None,NFEAT,NFEAT,NDIM), chunks=(1,NFEAT,NFEAT,NDIM), dtype='uint8')
    fres.create_dataset('disc_features_recon', (0,NFEAT,NFEAT,NDIM), maxshape=(None,NFEAT,NFEAT,NDIM), chunks=(1,NFEAT,NFEAT,NDIM), dtype='uint8')
    fres.create_dataset('disc_features_resid', (0,NFEAT,NFEAT,NDIM), maxshape=(None,NFEAT,NFEAT,NDIM), chunks=(1,NFEAT,NFEAT,NDIM), dtype='uint8')


print("Loading data")
reals = lib.datautils.load(result_fn, dataset='reals')
recons = lib.datautils.load(result_fn, dataset='reconstructed')
indices_now = np.arange(reals.len())

print("Initializing generator") 
reals_gen = lib.datautils.DataGenerator(reals, y=indices_now, batch_size=BATCH_SIZE, luptonize=False, shuffle=False, starti=startcount, once=True)
recons_gen = lib.datautils.DataGenerator(recons, y=indices_now, batch_size=BATCH_SIZE, luptonize=False, shuffle=False, starti=startcount, once=True)
print(f'Num to detect: {len(reals)}')

def resize_datasets(f, addsize):
    for dataset in new_datasets:
        f[dataset].resize(f[dataset].shape[0]+addsize, axis=0)

start = time.time()
moredata = True
nbatches = 0
count = startcount
while moredata:
    #fres = h5py.File(result_fn,"a")
    
    with tf.Session() as sess:
        s0 = time.time() 
        sess.run(tf.global_variables_initializer())
        print(f'Batch {nbatches}, count {count}')
        nbatches += 1
        
        print('getting images')        
        _reals, _indices_now = reals_gen.next()
        _recons, _indices_now = recons_gen.next()
        nimages = len(_reals)
        if reals_gen.is_done:
            moredata = False
        
        print('resizing datasets')
        resize_datasets(fres, nimages)
        
        #pad
        if nimages<BATCH_SIZE:
            _reals_padded = np.zeros((BATCH_SIZE, IMAGE_DIM))
            _reals_padded[:nimages] = _reals
            _reals = _reals_padded
            _recons_padded = np.zeros((BATCH_SIZE, IMAGE_DIM))
            _recons_padded[:nimages] = _recons
            _recons = _recons_padded

        print('running')
        feed_dict={real: _reals, reconstructed: _recons}
        _disc_real, _disc_recon, _disc_resid = sess.run(
                [disc_real, disc_recon, disc_resid],
                feed_dict=feed_dict)
        
        _disc_real = _disc_real.reshape((-1, NDIM, NFEAT, NFEAT)).transpose(0,2,3,1)
        _disc_real = (255.*_disc_real).astype('uint8')
        _disc_recon = _disc_recon.reshape((-1, NDIM, NFEAT, NFEAT)).transpose(0,2,3,1)
        _disc_recon = (255.*_disc_recon).astype('uint8')
        _disc_resid = _disc_resid.reshape((-1, NDIM, NFEAT, NFEAT)).transpose(0,2,3,1)
        _disc_resid = (255.*_disc_resid).astype('uint8')
        
        print('saving')
        for bb in range(nimages):
            fres['disc_features_real'][count, ...] = _disc_real[bb]
            fres['disc_features_recon'][count, ...] = _disc_recon[bb]
            fres['disc_features_resid'][count, ...] = _disc_resid[bb]
            count += 1

        e0 = time.time()
        print(f't iter: {e0-s0}')
                        
fres.close() 

end = time.time()
print(f"Time for {len(reals)} images: {end-start} s")

print("Done")
