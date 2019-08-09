# **************************************************
# * File Name : investigate_anomalies.py
# * Creation Date : 2019-08-06
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np
import pandas as pd

import plotter


imtag = 'i20.0_norm_100k'
gantag = '_features0.05go'

idx_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{imtag}_idx.npy'
results_dir = '/scratch/ksf293/kavli/anomaly/results'
scores_fn = f'{results_dir}/results_{imtag}{gantag}.npy'
plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-05'

cat_fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5.csv"

scores = np.load(scores_fn, allow_pickle=True)[:,4]
idxs = np.load(idx_fn)
cat = pd.read_csv(cat_fn) 
print(len(scores), len(cat), len(idxs))
print(list(cat.columns)) 
#for i in range(len(scores)):
scores_big = [scores[i] for i in range(len(scores)) if cat['i_extendedness_value'].iloc[idxs[i]]==1]
scores_small = [scores[i] for i in range(len(scores)) if cat['i_extendedness_value'].iloc[idxs[i]]!=1]
print(len(scores_big))
print(len(scores_small))
plotter.plot_dist([scores_big, scores_small], f'{plot_dir}/dist_{imtag}{gantag}_extendedness.png', labels=['extended', 'compact'])
#plotter.plot_dist(scores_small, f'{plot_dir}/dist_{imtag}{gantag}_compact.png')
