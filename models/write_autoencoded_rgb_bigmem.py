# **************************************************
# * File Name : write_autoencoded.py
# * Creation Date : 2019-08-07
# * Created By : kstoreyf
# * Description :
# **************************************************
import gc
import h5py
import os
import sys
sys.path.append(os.getcwd())
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
import time
from numba import cuda

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
print("Tensorflow version:", tf.__version__)

import tflib as lib
import tflib.datautils


DIM = 64
NSIDE = 96
NBANDS = 3
IMAGE_DIM = NSIDE*NSIDE*NBANDS
BATCH_SIZE = 1000

decode_latent = True
#decode_latent = False
start_nbatch = 0

#tag = 'gri_lambda0.3'
#tag = 'gri_100k_lambda0.3'
tag = 'gri_lambda0.3_3sigd'
aenum = 30000
#mode = 'residuals'
mode = 'reals'
#aetag = f'_latent32_{mode}'
aetag = f'_latent64_{mode}_long'
#aetag = '_aetest'
savetag = f'_model{aenum}{aetag}_decodetest'
base_dir = '/scratch/ksf293/anomalies'
results_dir = f'{base_dir}/results'
#imarr_fn = f'{base_dir}/data/images_h5/images_{tag}.h5'
save_fn = f'{results_dir}/autoencodes/autoencoded_{tag}{savetag}.npy'
if decode_latent:
    save_decode_fn = f'{results_dir}/decodes/decoded_{tag}{savetag}.npy'
results_fn = f'{results_dir}/results_{tag}.h5'
results_ae_fn = f'{results_dir}/results_ae_{tag}.h5'

ae_fn = f'{base_dir}/training_output/autoencoder_training/autoencoder_{tag}{aetag}/model-autoencoder-{aenum}'

#get data
print("Loading data", flush=True)
#print("WRITING AUTENCODES FOR REALS (NOT RESIDUALS)")
data = lib.datautils.load(results_fn, dataset=mode)
idxs = lib.datautils.load(results_fn, dataset='idxs')
object_ids = lib.datautils.load(results_fn, dataset='object_ids')
scores = lib.datautils.load(results_fn, dataset='disc_scores_sigma')
y = range(len(data))
data_gen = lib.datautils.DataGenerator(data, y=y, batch_size=BATCH_SIZE, shuffle=False, once=True,
                                        luptonize=False, normalize=False, smooth=False)

count = 0
ndata = len(data)
if not os.path.isfile(results_ae_fn):
    print(f"Making new result file at {results_ae_fn}")
    fres = h5py.File(results_ae_fn,"w")
    fres.create_dataset('idxs', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.create_dataset('object_ids', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.create_dataset('decoded', (ndata,NSIDE,NSIDE,NBANDS), maxshape=(ndata,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    fres.create_dataset('latents', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.create_dataset('ae_anomaly_scores', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.attrs['count'] = 0
    fres.close()
else:
    fres = h5py.File(results_ae_fn,"r")
    count = fres.attrs['count']
    fres.close()
    print(f"Loaded result file at {results_ae_fn}, count = {count}")


nbatch = start_nbatch

latents = []
#decodeds = []
start = time.time()
while not data_gen.is_done:
    # Because of memory issues, instantiate a new module for each batch
    # (Yes shouldn't have to do this but can't figure out another solution!)
    s0 = time.time()
    fres = h5py.File(results_ae_fn,"a")
    AutoEncoder = hub.Module(ae_fn)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print(f'Batch {nbatch}', flush=True)
        _data, _y = data_gen.next()
        _idx = idxs[list(_y)]
        _scores = scores[list(_y)]
        _object_id = object_ids[list(_y)]

        _latent_tensor = AutoEncoder(_data, signature='latent')
        _latent = sess.run(_latent_tensor)
        for i in range(len(_data)):
            latents.append([_latent[i], int(_idx[i]), _scores[i]])
            #fres['idxs'][count] = _idx[i]
            #fres['object_ids'][count] = _object_id[i]
            #fres['latents'][count] = _latent[i]

    tf.keras.backend.clear_session()

    if decode_latent:
        AutoEncoder = hub.Module(ae_fn)
        #fres = h5py.File(results_ae_fn,"a")
        with tf.Session() as sess:
            print("Decoding")
            # input _latent is a batch of latent-space representations
            #_decode_latent_tensor = AutoEncoder(_latent, signature='decode_latent')
            #_data = _data.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            #_decode_latent_tensor = AutoEncoder(_data.reshape((-1,IMAGE_DIM)))
            sess.run(tf.global_variables_initializer())

            _data, _y = data_gen.next()
            _idx = idxs[list(_y)]
            _decode_latent_tensor = AutoEncoder(_data)
            _decoded = sess.run(_decode_latent_tensor)
            _decoded = _decoded.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            _decoded = (255.*_decoded).astype('uint8')
            _data = _data.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
            _data = (255.*_decoded).astype('uint8')
            for i in range(len(_data)):
                #decodeds.append([_decoded[i], _idx[i], _scores[i]])
                residual = np.abs(np.subtract(_data[i], _decoded[i]))
                ae_anomaly_score = np.sum(residual)
                fres['idxs'][count] = _idx[i]
                fres['object_ids'][count] = _object_id[i]
                fres['latents'][count] = _latent[i]
                fres['reals'][count, ...] = _data[i]
                fres['decodeds'][count, ...] = _decoded[i]
                fres['residuals'][count, ...] = residual
                fres['ae_anomaly_scores'][count] = ae_anomaly_score
                fres.attrs['count'] = count+1
                count += 1
        #fres.close()
        tf.keras.backend.clear_session()

            #print('saving')
            #for bb in range(nimages):
            #idx = idxs[_indices_now[bb]]
            #objid = object_ids[_indices_now[bb]]
            #fres['idxs'][count] = idx
            #fres['object_ids'][count] = objid
            #fres['reconstructed'][count, ...] = _reconstructed[bb]
            #fres['gen_scores'][count] = _residual[bb]
            #fres.attrs['count'] = count+1
    
    fres.close()
    e0 = time.time()
    print(f"Time for batch: {e0-s0} s", flush=True)

    nbatch += 1
    n = gc.collect()
    print("Number of unreachable objects collected: ", n)
    # Clear the memory of each batch session

end = time.time()
np.save(save_fn, latents)
#if decode_latent:
#    np.save(save_decode_fn, decodeds)
print("Saved")
print(f"Time for {len(data)} images: {end-start} s")    



