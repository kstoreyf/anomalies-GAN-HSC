# **************************************************
# * File Name : loss.py
# * Creation Date : 2019-08-09
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

import save_images as saver


tag = 'i20.0_norm_100k_features0.05go'
#embtag = '_anoms'
#embtag = '_latent64_clust'
#embtag = '_anoms'
savetag = ''
anoms = True
if anoms:
    savetag += '_anoms'
#emb_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{embtag}.npy'
#embed = np.load(emb_fn, allow_pickle=True)

plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-08c'
plot_fn = f'{plot_dir}/residuals_{tag}{savetag}.png'

results_dir = '/scratch/ksf293/kavli/anomaly/results'
#auto_fn = f'{results_dir}/autoencoded_{tag}.npy'
results_fn = f'{results_dir}/results_{tag}.npy'
print("Loading results")
#res = np.load(results_fn, allow_pickle=True)

def get_results(results_fn, anoms=False):
    print("Loading data")
    res = np.load(results_fn, allow_pickle=True)
    if anoms:
        scores = res[:,4]
        mean = np.mean(scores)
        std = np.std(scores)
        anomidx = np.array([i for i in range(len(scores)) if scores[i]>mean+2*std])
        res = res[anomidx]
    images = res[:,0]
    images = np.array([np.array(im) for im in images])
    print(images.shape)
    images = images.reshape((-1, 96*96))
    recons = res[:,1]
    recons = np.array([np.array(im) for im in recons])
    recons = recons.reshape((-1, 96*96))
    resid_gen = res[:,2]
    resid_disc = res[:,3]
    scores = res[:,4]
    idxs = res[:,5]
    residuals = abs(images-recons)
    return images, residuals, idxs, scores, resid_gen, resid_disc

images, residuals, idxs, scores, resid_gen, resid_disc = get_results(results_fn, anoms=anoms)

plt.scatter(resid_gen, resid_disc, c=scores, marker='.', s=8)
plt.xlabel("generator residual")
plt.ylabel("discriminator residual")
cbar = plt.colorbar()
cbar.set_label('anomaly score', rotation=270)
plt.savefig(plot_fn)
