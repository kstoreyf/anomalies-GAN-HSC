# **************************************************
# * File Name : view_results.py
# * Creation Date : 2019-07-30
# * Last Modified :
# * Created By : 
# * Description :
# **************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

#tag = 'i85k_96x96_norm'
tag = 'i20.0_norm_try3'
results_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/results'
results_fn = f'{results_dir}/results_{tag}.npy'

save_fn = f'{results_dir}/results_{tag}.png'

res = np.load(results_fn, allow_pickle=True)
reals = res[:,0]
recons = res[:,1]
scores = res[:,2]
idx = np.argsort(scores)
scores = scores[idx]
reals = reals[idx]
recons = recons[idx]



nrows = 4
ncols = 8

step = int(len(scores)/(nrows*ncols))
reals = reals[::step]
recons = recons[::step]
scores = scores[::step]

fig, axarr = plt.subplots(nrows,ncols, figsize=(12,12))
plt.subplots_adjust(hspace=0.2, wspace=0.2)
cc = 0
for i in range(nrows):
    for j in range(ncols):
        real = reals[cc].reshape((96,96))
        recon = recons[cc].reshape((96,96))
        combined = np.vstack((recon, real))
        #print(combined.shape)
        axarr[i][j].imshow(combined, cmap='gray', origin='lower')
        axarr[i][j].set_xticks([])
        axarr[i][j].set_yticks([])
        axarr[i][j].set_title(scores[cc])
        cc += 1 

plt.savefig(save_fn)

