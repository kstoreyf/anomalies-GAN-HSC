# **************************************************
# * File Name : write_decoded_rgb.py
# * Creation Date : 2021-07-08
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

start_nbatch = 0
#start_count = 739000
start_count = 0
overwrite = True

tag = 'gri_lambda0.3'
#tag = 'gri_1k_lambda0.3'
#tag = 'gri_lambda0.3_3sigd'
aenum = 30000
#aenum = 1000
#mode = 'residuals'
mode = 'reals'
aetag = f'_latent64_{mode}_long_lr1e-4'
savetag = f'_model{aenum}{aetag}'
base_dir = '/scratch/ksf293/anomalies'
results_dir = f'{base_dir}/results'
results_fn = f'{results_dir}/results_{tag}.h5'
results_ae_fn = f'{results_dir}/results_aefull_{tag}{savetag}.h5'

ae_fn = f'{base_dir}/training_output/autoencoder_training/autoencoder_{tag}{aetag}/model-autoencoder-{aenum}'

#get data
print("Loading data", flush=True)
data = lib.datautils.load(results_fn, dataset=mode)
idxs = lib.datautils.load(results_fn, dataset='idxs')
object_ids = lib.datautils.load(results_fn, dataset='object_ids')
y = range(len(data))
data_gen = lib.datautils.DataGenerator(data, y=y, batch_size=BATCH_SIZE, shuffle=False, once=True,
                                        luptonize=False, normalize=False, smooth=False)

count = 0
ndata = len(data)
if not os.path.isfile(results_ae_fn):
    print(f"Making new result file at {results_ae_fn}", flush=True)
    fres = h5py.File(results_ae_fn,"w")
    fres.create_dataset('idxs', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.create_dataset('object_ids', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,), dtype='uint64')
    fres.create_dataset('decodeds', (ndata,NSIDE,NSIDE,NBANDS), maxshape=(ndata,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    fres.create_dataset('reals', (ndata,NSIDE,NSIDE,NBANDS), maxshape=(ndata,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    fres.create_dataset('residuals', (ndata,NSIDE,NSIDE,NBANDS), maxshape=(ndata,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    fres.create_dataset('latents', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.create_dataset('ae_anomaly_scores', (ndata,), maxshape=(ndata,), chunks=(BATCH_SIZE,))
    fres.attrs['count'] = 0
    fres.close()
else:
    fres = h5py.File(results_ae_fn,"r")
    count = fres.attrs['count']
    fres.close()
    print(f"Loaded result file at {results_ae_fn}, count = {count}", flush=True)

if overwrite:
    count = start_count

nbatch = start_nbatch

n_so_far = 0
start = time.time()
while not data_gen.is_done:
    # Because of memory issues, instantiate a new module for each batch
    # (Yes shouldn't have to do this but can't figure out another solution!)
    print(f"Batch {nbatch}, count {count}", flush=True)
    s0 = time.time()
    fres = h5py.File(results_ae_fn,"a")

    _data, _y = data_gen.next()
    _idx = idxs[list(_y)]
    _object_id = object_ids[list(_y)]

    n_so_far += len(_data)
    print(f"n_so_far: {n_so_far}, start_count: {start_count}", flush=True)
    if n_so_far <= start_count:
        print("not at start yet! skipping", flush=True)
        fres.close()
        nbatch += 1
        #count += BATCH_SIZE
        continue

    AutoEncoder = hub.Module(ae_fn)
    with tf.Session() as sess:
        print(f"Decoding, starting at count = {count}", flush=True)
        sess.run(tf.global_variables_initializer())

        #_data, _y = data_gen.next()
        #_idx = idxs[list(_y)]
        #_object_id = object_ids[list(_y)]

        _decode_latent_tensor = AutoEncoder(_data)
        _decoded = sess.run(_decode_latent_tensor)

        _decoded = _decoded.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
        #_decoded = _decoded.astype('uint8')
        _data = _data.reshape((-1, NBANDS, NSIDE, NSIDE)).transpose(0,2,3,1)
        for i in range(len(_data)):
            
            residual = np.abs(np.subtract(_data[i], _decoded[i]))
            # Divide by 255 to be consistent with generator score definition
            ae_anomaly_score = np.sum(residual)/255. 
            residual = residual.astype('uint8')
            decoded = _decoded[i].astype('uint8')
            fres['idxs'][count] = _idx[i]
            fres['object_ids'][count] = _object_id[i].astype('uint64')
            #fres['latents'][count] = _latent[i]
            fres['reals'][count, ...] = _data[i]
            fres['decodeds'][count, ...] = decoded
            fres['residuals'][count, ...] = residual
            fres['ae_anomaly_scores'][count] = ae_anomaly_score
            fres.attrs['count'] = count+1
            count += 1

    tf.keras.backend.clear_session()
    fres.close()
    e0 = time.time()
    print(f"Time for batch: {e0-s0} s", flush=True)

    nbatch += 1
    n = gc.collect()
    print("Number of unreachable objects collected: ", n, flush=True)
    # Clear the memory of each batch session

end = time.time()
print("Saved")
print(f"Time for {len(data)} images: {end-start} s")    



