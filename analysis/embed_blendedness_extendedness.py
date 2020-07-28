# **************************************************
# * File Name : investigate_anomalies.py
# * Creation Date : 2019-08-06
# * Created By : kstoreyf
# * Description :
# **************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py

import utils
import plotter


#tag = 'gri_cosmos'
tag = 'gri_3sig'
base_dir = '..'

aenum = 29500
#aetag = '_latent16'
aetag = '_latent32'
savetag = f'_model{aenum}{aetag}'

results_dir = f'{base_dir}/results'
results_fn = f'{results_dir}/results_{tag}.h5'

save_fn = f'{base_dir}/results/embeddings/embedding_blendextend_{tag}{savetag}.npy'
plot_dir = f'{base_dir}/plots/plots_2020-07-08'
plot_fn = f'{plot_dir}/blendextend_{tag}{savetag}.png'

cat_fn = f'{base_dir}/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'

print("Loading data")
res = h5py.File(results_fn, 'r')
scores = res['anomaly_scores']
idxs = [int(idx) for idx in res['idxs'][:]]

print("Loading catalog")
cat = pd.read_csv(cat_fn) 

print("Getting blend & extend")
extend = [cat['i_cmodel_ellipse_radius'].iloc[idx] for idx in idxs]
extend = [np.log10(e) for e in extend]
blend = [cat['i_blendedness_abs_flux'].iloc[idx] for idx in idxs]
#fix bad blends
blend = [b if b>=0 else 0 for b in blend]

result = np.array([extend, blend, scores, idxs])
np.save(save_fn, result)

print("Plotting")
plt.scatter(extend, blend, marker='.', c=scores, cmap='viridis', s=8,
                                                       vmin=min(scores), vmax=4000)
plt.xlabel('log(extendedness)')
plt.ylabel('blendedness')

cbar = plt.colorbar(extend='max')
cbar.set_label('anomaly score', rotation=270, labelpad=10)
#plt.xlim(0, 11)
#plt.ylim(-0.1, 1.1)
#plt.xscale("log")
plt.savefig(plot_fn)
