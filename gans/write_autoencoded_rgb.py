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
import tflib as lib
import tflib.datautils


DIM = 64
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 1000

decode_latent = True
startcount = 0

#tag = 'gri_100k'
#tag = 'gri_cosmos'
tag = 'gri_3sig'
#aenum = 29500
aenum = 29500
#aetag = '_aereal'
aetag = '_latent32_real'
#aetag = '_latent32'
#aetag = '_aetest'
savetag = f'_model{aenum}{aetag}'
results_dir = '/scratch/ksf293/kavli/anomaly/results'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
save_fn = f'{results_dir}/autoencodes/autoencoded_{tag}{savetag}.npy'
if decode_latent:
    save_decode_fn = f'{results_dir}/decodes/decoded_{tag}{savetag}.npy'
results_fn = f'{results_dir}/results_{tag}.h5'

ae_fn = f'/scratch/ksf293/kavli/anomaly/training_output/autoencoder_{tag}{aetag}/model-autoencoder-{aenum}'
AutoEncoder = hub.Module(ae_fn)

#get data
print("Loading data")
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(
                                                    results_fn, imarr_fn)
residuals, reals, recons = utils.get_residuals(reals, recons) # this luptonizes the reals so we don't have to
print(residuals.shape)

#data = residuals
print("WRITING AUTENCODES FOR REALS (NOT RESIDUALS)")
data = reals
y = range(len(data))
data_gen = lib.datautils.DataGenerator(data, y=y, batch_size=BATCH_SIZE, shuffle=False, once=True,
                                        luptonize=False, normalize=False, smooth=False)

count = startcount

latents = []
decodeds = []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    start = time.time()
    while not data_gen.is_done:
        print(f'Batch {count}')
        _data, _y = data_gen.next()
        _idx = idxs[_y]
        _scores = scores[_y]

        s0 = time.time()
        _latent_tensor = AutoEncoder(_data, signature='latent')
        _latent = sess.run(_latent_tensor)
        for i in range(len(_data)):
            latents.append([_latent[i], int(_idx[i]), _scores[i]])

        e0 = time.time()
        print(f"Time for batch: {e0-s0} s")
        
        if decode_latent:
            print("Decoding")
            # input _latent is a batch of latent-space representations
            #_decode_latent_tensor = AutoEncoder(_latent, signature='decode_latent')
            #_data = _data.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            #print(_data[0])
            #print(_data.reshape((-1,IMAGE_DIM))[0])
            _decode_latent_tensor = AutoEncoder(_data.reshape((-1,IMAGE_DIM)))
            _decoded = sess.run(_decode_latent_tensor)
            #print(_decoded[0])
            _decoded = _decoded.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            #print(_decoded[0])
            #_decoded = (255.*_decoded).astype('uint8')
            for i in range(len(_data)):
                decodeds.append([_decoded[i], _idx[i], _scores[i]])

        count += 1
         
    end = time.time()
    np.save(save_fn, latents)
    if decode_latent:
        np.save(save_decode_fn, decodeds)
    print("Saved")
    print(f"Time for {len(data)} images: {end-start} s")    



