# **************************************************
# * File Name : write_autoencoded.py
# * Creation Date : 2019-08-07
# * Created By : kstoreyf
# * Description :
# **************************************************
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


DIM = 64
NSIDE = 96
IMAGE_DIM = NSIDE*NSIDE
BATCH_SIZE = 1000

startcount = 0
tag = 'i20.0_norm_100k_features0.05go'
aenum = 9000
aetag = '_latent64'
savetag = aetag+''
results_dir = '/scratch/ksf293/kavli/anomaly/results'
save_fn = f'{results_dir}/autoencoded_{tag}{savetag}.npy'
results_fn = f'{results_dir}/results_{tag}.npy'

ae_fn = f'/scratch/ksf293/kavli/anomaly/training_output/autoencoder_{tag}{aetag}/model-autoencoder-{aenum}'
AutoEncoder = hub.Module(ae_fn)

def get_anomalous_idxs(scores, sigma=2):
    mean = np.mean(scores)
    std = np.std(scores)
    return np.array([i for i in range(len(scores)) if scores[i]>mean+sigma*std])

def get_residuals(res):
    images = res[:,0]
    images = np.array([np.array(im) for im in images])
    images = images.reshape((-1, NSIDE, NSIDE))
    recons = res[:,1]
    recons = np.array([np.array(im) for im in recons])
    recons = recons.reshape((-1, NSIDE, NSIDE))
    return abs(images-recons)


#get data
print("Loading data")
#res = lib.datautils.load_numpy(results_fn)
res = np.load(results_fn, allow_pickle=True)
scores = res[:,4]
res = res[get_anomalous_idxs(scores)]
data = get_residuals(res)
scores = res[:,4]
idxs = res[:,5]
print(data.shape)

moredata = 1
count = startcount
loc = startcount*BATCH_SIZE

latents = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    while moredata:
        print(f'Batch {count}')
        _data = data[loc:loc+BATCH_SIZE].reshape((-1, IMAGE_DIM))
        _idx = idxs[loc:loc+BATCH_SIZE]
        _scores = scores[loc:loc+BATCH_SIZE]
        #if (len(idx))<BATCH_SIZE:
        s0 = time.time()
        #for i in range(len(data)):
        #_data = data[i].reshape((-1, IMAGE_DIM))
        _latent_tensor = AutoEncoder(_data, signature='latent')
        _latent = sess.run(_latent_tensor)
        for i in range(len(_data)):
            latents.append([_latent[i], _idx[i], _scores[i]])

        e0 = time.time()
        print(f"Time for batch: {e0-s0} s")
        
        loc += BATCH_SIZE
        if loc>=len(data):
            moredata = 0
        count += 1
         
    end = time.time()
    np.save(save_fn, latents)
    print("Saved")
    print(f"Time for {len(data)} images: {end-start} s")    



