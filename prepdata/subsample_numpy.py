# **************************************************
# * File Name : subsample_numpy.py
# * Creation Date : 2019-07-30
# * Created By : kstoreyf
# * Description: Creates random subsample of numpy array 
# **************************************************

import numpy as np

maintag = 'i20.0'
tag = 'i20.0_norm'

nsample = 100000
out_tag = '100k'

print("Loading array")
imdir = '/scratch/ksf293/kavli/anomaly/data/images_np'
imarr_fn = f'{imdir}/imarr_{tag}.npy'
idx_fn = f'{imdir}/imarr_{maintag}_idx.npy'
arr = np.load(imarr_fn)
idx = np.load(idx_fn)

print("Sumbsampling")
rands = np.random.choice(len(arr), size=nsample, replace=False)
subarr = arr[rands]
subidx = idx[rands]

print("Saving")
np.save(f'{imdir}/imarr_{tag}_{out_tag}.npy', subarr)
np.save(f'{imdir}/imarr_{tag}_{out_tag}_idx.npy', subidx)
print("Saved")
