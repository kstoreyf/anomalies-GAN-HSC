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

import utils


DIM = 64
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 1000

decode_latent = True
startcount = 0
tag = 'gri_cosmos_fix'
#tag = 'i20.0_norm_100k_features0.05go'
aenum = 9500
#aenum = 9000
#aetag = '_aetest'
aetag = '_latent16'
savetag = f'_model{aenum}{aetag}'

results_dir = '/scratch/ksf293/kavli/anomaly/results'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
save_fn = f'{results_dir}/autoencodes/autoencoded_{tag}{savetag}.npy'
if decode_latent:
    save_decode_fn = f'{results_dir}/autoencodes/decoded_{tag}{savetag}.npy'
results_fn = f'{results_dir}/results_{tag}.h5'

ae_fn = f'/scratch/ksf293/kavli/anomaly/training_output/autoencoder_{tag}{aetag}/model-autoencoder-{aenum}'
AutoEncoder = hub.Module(ae_fn)

#get data
print("Loading data")
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(
                                                    results_fn, imarr_fn)
residuals, reals, recons = utils.get_residuals(reals, recons)

#res = lib.datautils.load_numpy(results_fn)
#res = np.load(results_fn, allow_pickle=True)
#scores = res[:,4]
#res = res[get_anomalous_idxs(scores)]
#data = get_residuals(res)
#scores = res[:,4]
#idxs = res[:,5]
#print(data.shape)
data = residuals

moredata = 1
count = startcount
loc = startcount*BATCH_SIZE

latents = []
decodeds = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    while moredata:
        print(f'Batch {count}')
        # TODO: use data generator in libutils (see e.g. wganrgb.py)
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
        
        print("Decoding")
        if decode_latent:
            # input _latent is a batch of latent-space representations
            _decode_latent_tensor = AutoEncoder(_latent, signature='decode_latent')
            _decoded = sess.run(_decode_latent_tensor)
            _decoded = _decoded.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            _decoded = (255.*_decoded).astype('uint8')
            for i in range(len(_data)):
                decodeds.append([_decoded[i], _idx[i], _scores[i]])

        loc += BATCH_SIZE
        if loc>=len(data):
            moredata = 0
        count += 1
         
    end = time.time()
    np.save(save_fn, latents)
    if decode_latent:
        np.save(save_decode_fn, decodeds)
    print("Saved")
    print(f"Time for {len(data)} images: {end-start} s")    



