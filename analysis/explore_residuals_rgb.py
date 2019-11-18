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
import utils


tag = 'gri'
#embtag = '_anoms'
#embtag = '_latent64_clust'
#embtag = '_anoms'
#savetag = '_fix_weight0.2'
#savetag = '_max4000'
savetag = ''
sigma = 0
if sigma:
    savetag += f'_{sigma}sigma'
#emb_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{embtag}.npy'
#embed = np.load(emb_fn, allow_pickle=True)

plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-14'
plot_fn = f'{plot_dir}/residuals_{tag}{savetag}.png'

results_dir = '/scratch/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.h5'

imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

#images, residuals, idxs, scores, resid_gen, resid_disc = utils.get_results(
#                                                    results_fn, imarr_fn, n_anoms=0)
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(
                                                    results_fn, imarr_fn, sigma=sigma)

gen_scores = np.array(gen_scores)
disc_scores = np.array(disc_scores)
anomaly_weight = 0.05
anomaly_score = (1-anomaly_weight)*gen_scores + anomaly_weight*disc_scores
plt.scatter(gen_scores, disc_scores, c=anomaly_score, marker='.', s=8, alpha=1, vmax=6000)

plt.xlabel("generator residual")
plt.ylabel("discriminator residual")

#plt.xlim(2000,4000)
#plt.xlim(0,4000)
#plt.ylim(0,22000)
cbar = plt.colorbar(extend='max')
cbar.set_label('anomaly score', rotation=270, labelpad=10)
plt.savefig(plot_fn)
