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
import plotter_rgb as plotter
import utils

NBANDS = 3

tag = 'gri'
#embtag = '_anoms'
sigma = 3
embtag = f'_clust_residuals_{sigma}sigma'
savetag = embtag+'_ncomp10'
n_components = 10
clusteron = 'embed'

emb_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{embtag}.npy'
embed = np.load(emb_fn, allow_pickle=True)

plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-14b'
plot_fn = f'{plot_dir}/gmm_{tag}{savetag}.png'

results_dir = '/scratch/ksf293/kavli/anomaly/results'
auto_fn = f'{results_dir}/autoencoded_{tag}.npy'
results_fn = f'{results_dir}/results_{tag}.h5'

imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

#print("Loading results")
#res = np.load(results_fn, allow_pickle=True)

def get_autoencoded(auto_fn):
    auto = np.load(auto_fn, allow_pickle=True)
    latents = auto[:,0]
    print(latents.shape)
    idxs = auto[:,1]
    scores = auto[:,2]
    latents = latents.tolist()
    return latents, idxs, scores

n_anoms = 0
#if sigma>0:
#   savetag += f'_{sigma}sigma'
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(results_fn, imarr_fn, n_anoms=n_anoms, sigma=sigma)
residuals, reals, recons = utils.get_residuals(reals, recons)

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
    #idxs = embed[3]
    #idxs = [int(idx) for idx in idxs]
    #reals = reals[idxs]
    #residuals = residuals[idx]
    #images, residuals = get_images(res)
    print(values.shape)

#n_components = 2
print("Making GMM")
gmm = GMM(n_components=n_components).fit(values)
print("Predicting")
labels = gmm.predict(values)
#plt.scatter(X[:, 0], X[:, 1], c=labels, s=40, cmap='viridis')
plt.scatter(embed[0], embed[1], c=labels, s=10, cmap='rainbow')
plt.savefig(plot_fn)

def make_batch(arr, save_fn, n=128):
    arr = np.array(arr)
    ims = arr[:n]
    ims.reshape((-1, 96, 96, NBANDS))
    saver.save_images(ims, save_fn)


res = {'reconstructed': recons, 'gen_scores': gen_scores, 'disc_scores': disc_scores,
       'anomaly_scores': scores, 'idxs': idxs, 'object_ids': object_ids}
imarr = {'images':reals}

print("writing clusters")
for nc in range(n_components):
    ims = np.array([reals[i] for i in range(len(labels)) if labels[i]==nc])
    n = 25
    if len(ims)<n:
        n = len(ims)
    sample_idx = [int(r) for r in np.random.choice(len(ims), size=n, replace=False)]
    make_batch(ims[sample_idx], f'{plot_dir}/gmmcluster_{tag}{savetag}-{nc}.png')
    resids = np.array([residuals[i] for i in range(len(labels)) if labels[i]==nc])
    print(nc)
    print(len(resids))
    make_batch(resids[sample_idx], f'{plot_dir}/gmmcluster_{tag}{savetag}-{nc}-residuals.png')
    #res_sample = np.array([res[i] for i in range(len(labels)) if labels[i]==nc])

    idx_clust = [i for i in range(len(labels)) if labels[i]==nc]
    idx_clust_sample = [int(r) for r in np.random.choice(idx_clust, size=24, replace=False)]

    plotter.plot_comparisons(res, imarr, f'{plot_dir}/comp_gmmcluster_{tag}{savetag}-{nc}.png', which='sample', sample=idx_clust_sample, luptonize=False)

