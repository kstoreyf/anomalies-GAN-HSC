# ********************************************************
# * File Name : combine_result_arrays.py
# * Creation Date : 2019-08-05
# * Created By : kstoreyf
# * Description : Combines batched numpy arrays into one 
#       large numpy array of anomaly detection results
# ********************************************************

import os
import numpy as np

tag = 'i20.0_norm_features0.05go'
#tag = 'i20.0_norm_100k_features0.05go'
res_dir = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}'
savetag = '_scores'
save_fn = f'/scratch/ksf293/kavli/anomaly/results/results_{tag}{savetag}.npy'


print("Combining numpy arrays...")
#arrs = np.concatenate([np.load(f'{res_dir}/{fn}', allow_pickle=True) for fn in os.listdir(res_dir)])
nfiles = 942

scores_all = None
#for fn in os.listdir(res_dir):
for batchnum in range(nfiles):
    if batchnum % 10 == 0:
        print(batchnum)
    res = np.load(f'{res_dir}/results_{tag}-{batchnum}.npy', allow_pickle=True)
    scores = res[:,4]
    if scores_all is None:
        scores_all = scores
    else:
        scores_all = np.concatenate((scores_all, scores))

print("Loaded and concatenated")
np.save(save_fn, scores_all.astype('float32'))
print("Result arrays combined")
