# ********************************************************
# * File Name : combine_result_arrays.py
# * Creation Date : 2019-08-05
# * Created By : kstoreyf
# * Description : Combines batched numpy arrays into one 
#       large numpy array of anomaly detection results
# ********************************************************

import os
import numpy as np

tag = 'i20.0_norm_100k_features0.05go'
res_fn = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}.npy'
savetag = '_scores'
save_fn = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}{savetag}.npy'

print("Loading")
res = np.load(res_fn, allow_pickle=True)
scores = res[:,4]

print("Saving")
np.save(save_fn, scores.astype('float32'))
print("Saved")
