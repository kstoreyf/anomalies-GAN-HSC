# **************************************************
# * File Name : make_umap.py
# * Creation Date : 2019-08-02
# * Created By : kstoreyf
# * Description : Generates a UMAP of the given data.
# **************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import umap
import seaborn as sns

#tag = 'i20.0_norm_100k'
#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
tag = 'i20.0_norm_100k_features0.05go'
#results_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/results'
results_dir = '/scratch/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.npy'
byscore = True
anoms = True
#anoms = False

plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-01'
#savetag = '_10k'
savetag = ''
if anoms:
    savetag += '_anoms'
reducer = umap.UMAP()

print("Loading data")
res = np.load(results_fn, allow_pickle=True)
images = res[:,0]
images = np.array([np.array(im) for im in images])
print(images.shape)
images = images.reshape((-1, 96*96))
scores = res[:,4]
if anoms:
    mean = np.mean(scores)
    std = np.std(scores)    
    images = np.array([images[i] for i in range(len(images)) if scores[i]>mean+2*std])
    scores = np.array([scores[i] for i in range(len(scores)) if scores[i]>mean+2*std])
    print("Anoms:")
    print(images.shape)
print("Reducing")
reducer = umap.UMAP()
embedding = reducer.fit_transform(images)

print(f"Embedding shape: {embedding.shape}")

if byscore:
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='.', c=scores, cmap='viridis', s=8)
    cbar = plt.colorbar()
    cbar.set_label('anomaly score', rotation=270)
else:
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='.')
plt.gca().set_aspect('equal', 'datalim')
plt.savefig(f'{plot_dir}/umap_{tag}{savetag}.png')


