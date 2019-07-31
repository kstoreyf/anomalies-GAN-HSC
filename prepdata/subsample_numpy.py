# **************************************************
# * File Name : subsample_numpy.py
# * Creation Date : 2019-07-30
# * Created By : kstoreyf
# * Description: Creates random subsample of numpy array 
# **************************************************

import numpy as np

tag = 'i20.0'
nsample = 10000
out_tag = '10k'

print("Loading array")
imdir = '/scratch/ksf293/kavli/anomaly/data/images_np'
imarr_fn = f'{imdir}/imarr_{tag}.npy'
arr = np.load(imarr_fn)

print("Sumbsampling")
rands = np.random.choice(len(arr), size=nsample, replace=False)
subarr = arr[rands]

print("Saving")
np.save(f'{imdir}/imarr_{tag}_{out_tag}.npy', subarr)
print("Saved")
