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

def main():
    #tag = 'i20.0_norm_100k'
    #imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
    tag = 'i20.0_norm_100k_features0.05go'
    #results_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/results'
    results_dir = '/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.npy'
    savetag = ''
    
    #aetag = '_latent64'
    #savetag += aetag+'_clust'
    #auto_fn = f'{results_dir}/autoencoded_{tag}{aetag}.npy'
    
    #savetag += '_residdisc' 
    #savetag = '_clust'
    mode = 'residuals'
    #mode = 'images'
    savetag += f'_{mode}'
    savetag += '_clust_residdisc'

    anoms = True
    #savetag += '_residdisc'

    plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-08c'

    #savetag += f'_{mode}'
    if anoms:
        savetag += '_anoms'
    save_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{savetag}.npy'
    plot_fn = f'{plot_dir}/umap_{tag}{savetag}.png'
    
    #values, idxs, scores = get_autoencoded(auto_fn) 
    values, idxs, scores, resid_gen, resid_disc = get_results(results_fn, mode, anoms=anoms)
    embed(values, idxs, resid_disc, plot_fn, save_fn)

def get_autoencoded(auto_fn):
    auto = np.load(auto_fn, allow_pickle=True)
    print(auto.shape)
    latents = auto[:,0]
    print(latents.shape)
    idxs = auto[:,1]
    scores = auto[:,2]
    latents = latents.tolist() 
    return latents, idxs, scores

def get_results(results_fn, mode, anoms=False):
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
    if mode=='images':
        values = images
    if mode=='residuals':
        residuals = abs(images-recons)
        values = residuals
    return values, idxs, scores, resid_gen, resid_disc
    
    
def embed(values, idxs, colorby, plot_fn, save_fn):
    
    #values = np.array(values).flatten() 
    reducer = umap.UMAP(n_neighbors=10, min_dist=0.05)
    embedding = reducer.fit_transform(values)

    plt.scatter(embedding[:, 0], embedding[:, 1], marker='.', c=colorby, cmap='viridis', s=8)
    cbar = plt.colorbar()
    cbar.set_label('anomaly score', rotation=270)
    
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(plot_fn)
    
    result = np.array([embedding[:,0], embedding[:,1], colorby, idxs])
    np.save(save_fn, result)

if __name__=='__main__':
    main()
