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

import utils

def main():
   
    #tag = 'gri_100k'
    tag = 'gri_cosmos_fix'
    plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2020-07-08'

    aenum = 9500
    aetag = '_latent16'
    savetag = f'_model{aenum}{aetag}'
    
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    auto_fn = f'{results_dir}/autoencodes/autoencoded_{tag}{savetag}.npy'

    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

    savetag = ''
  
    #mode = 'images'
    #mode = 'residuals'
    mode = 'auto'
    savetag += f'_{mode}'

    #n_anoms = False
    #savetag += '_residdisc'
    n_anoms = 0
    sigma = 0
    if sigma>0:
        savetag += f'_{sigma}sigma'

    if n_anoms:
        savetag += '_anoms'
    save_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{savetag}.npy'
    plot_fn = f'{plot_dir}/umap_{tag}{savetag}.png'
    
    if mode=='auto':
        values, idxs, scores = get_autoencoded(auto_fn) 
    elif mode=='images' or mode=='residuals':
        reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(results_fn, imarr_fn, n_anoms=n_anoms, sigma=sigma)
    
    if mode=='images':
        values = reals    
    if mode=='residuals':
        residuals, _, _ = utils.get_residuals(reals, recons)
        values = residuals
    
    embed(values, idxs, scores, plot_fn, save_fn)

def get_autoencoded(auto_fn):
    auto = np.load(auto_fn, allow_pickle=True)
    print(auto.shape)
    latents = auto[:,0]
    print(latents.shape)
    idxs = auto[:,1]
    scores = auto[:,2]
    latents = latents.tolist() 
    return latents, idxs, scores

    
def embed(values, idxs, colorby, plot_fn, save_fn):
    print("Reshaping") 
    values = np.array(values)
    values = values.reshape((values.shape[0], -1))
    print(values.shape)
    print("Embedding")
    #values = np.array(values).flatten() 
    reducer = umap.UMAP(n_neighbors=5, min_dist=0.05)
    embedding = reducer.fit_transform(values)
    print("Plotting")
    plt.scatter(embedding[:, 0], embedding[:, 1], marker='.', c=colorby, cmap='viridis', s=8,
                                                                vmin=min(colorby), vmax=4000)
    plt.xlabel('umap 1')
    plt.ylabel('umap 2')
    
    cbar = plt.colorbar(extend='max')
    cbar.set_label('anomaly score', rotation=270, labelpad=10)
    #plt.clim(0,4000)
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(plot_fn)
    
    result = np.array([embedding[:,0], embedding[:,1], colorby, idxs])
    np.save(save_fn, result)

if __name__=='__main__':
    main()
