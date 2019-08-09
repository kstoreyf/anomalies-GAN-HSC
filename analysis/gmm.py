# **************************************************
# * File Name : gmm.py
# * Creation Date : 2019-08-08
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.mixture import GaussianMixture as GMM

import save_images as saver
import plotter

tag = 'i20.0_norm_100k_features0.05go'
#embtag = '_anoms'
embtag = '_latent64_clust'
#embtag = '_residuals_clust_anoms'
savetag = embtag+'_ncomp2'
n_components = 2
clusteron = 'embed'

emb_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{embtag}.npy'
embed = np.load(emb_fn, allow_pickle=True)

plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-08c'
plot_fn = f'{plot_dir}/gmm_{tag}{embtag}{savetag}.png'

results_dir = '/scratch/ksf293/kavli/anomaly/results'
auto_fn = f'{results_dir}/autoencoded_{tag}.npy'
results_fn = f'{results_dir}/results_{tag}.npy'
print("Loading results")
res = np.load(results_fn, allow_pickle=True)

def get_autoencoded(auto_fn):
    auto = np.load(auto_fn, allow_pickle=True)
    latents = auto[:,0]
    print(latents.shape)
    idxs = auto[:,1]
    scores = auto[:,2]
    latents = latents.tolist()
    return latents, idxs, scores


def get_images(res):
    images = res[:,0]
    images = np.array([np.array(im) for im in images])
    images = images.reshape((-1, 96*96))
    recons = res[:,1]
    recons = np.array([np.array(im) for im in recons])
    recons = recons.reshape((-1, 96*96))
    residuals = abs(images-recons)
    return images, residuals

print("Getting values")
#if 'auto' in embtag or 'latent' in embtag:
if clusteron=="latent":
    latents, idxs, scores = get_autoencoded(auto_fn)
    idxs = [int(idx) for idx in idxs]
    idxs = np.array(idxs)
    res = np.array(res)
    res = res[idxs]
    images, residuals = get_images(res)
    values = latents
elif clusteron=="embed":
    values = np.array([embed[0], embed[1]]).T
    idxs = embed[3]
    idxs = [int(idx) for idx in idxs]
    res = res[idxs]
    images, residuals = get_images(res)
    print(values.shape)

#n_components = 2
print("Making GMM")
gmm = GMM(n_components=n_components).fit(values)
print("Predicting")
labels = gmm.predict(values)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(embed[0], embed[1], c=labels, s=10, cmap='viridis')
plt.savefig(plot_fn)

def make_batch(arr, save_fn, n=128):
    arr = np.array(arr)
    ims = arr[:n]
    ims.reshape((-1, 96, 96))
    saver.save_images(ims, save_fn)

print("writing clusters")
for nc in range(n_components):
    ims = np.array([images[i] for i in range(len(labels)) if labels[i]==nc])
    n = 128
    sample_idx = [int(r) for r in np.random.choice(len(ims), size=n, replace=False)]
    make_batch(ims[sample_idx], f'{plot_dir}/gmmcluster_{tag}{savetag}-{nc}.png')
    resids = np.array([residuals[i] for i in range(len(labels)) if labels[i]==nc])
    print(nc)
    print(len(resids))
    make_batch(resids[sample_idx], f'{plot_dir}/gmmcluster_{tag}{savetag}-{nc}-residuals.png')
    res_sample = np.array([res[i] for i in range(len(labels)) if labels[i]==nc])
    plotter.plot_comparisons(res_sample, f'{plot_dir}/comp_gmmcluster_{tag}{savetag}-{nc}.png', which='random')

