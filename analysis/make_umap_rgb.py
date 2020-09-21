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
   
    #tag = 'gri_3sig'
    tag = 'gri_100k'
    #tag = 'gri_cosmos'
    base_dir = '/scratch/ksf293/kavli/anomaly'
    #base_dir = '..'
    #plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2020-07-08'
    make_plot = False
    plot_dir = f'../plots/plots_2020-07-08'
    savetag = ''

    aenum = 29500
    #aenum = 9500
    
    #aetag = '_latent32_real'
    aetag = '_latent16_real'
    #aetag = '_latent32'
    autotag = f'_model{aenum}{aetag}'

    
    results_dir = f'{base_dir}/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    auto_fn = f'{results_dir}/autoencodes/autoencoded_{tag}{autotag}.npy'

    imarr_fn = f'{base_dir}/data/images_h5/images_{tag}.h5'

    #mode = 'reals'
    mode = 'residuals'
    #mode = 'auto'
    
    # dataset choices
    n_anoms = 0
    sigma = 0
    # umap params
    n_neighbors = 5
    min_dist = 0.1

    if sigma>0:
        savetag += f'_{sigma}sigma'
    if n_anoms:
        savetag += '_anoms'
    if mode=='auto':
        savetag += autotag
    if n_neighbors and min_dist:
        savetag += f'_nn{n_neighbors}md{min_dist}'
    
    save_fn = f'{base_dir}/results/embeddings/embedding_umap_{mode}_{tag}{savetag}.npy'
    plot_fn = f'{plot_dir}/umap_{mode}_{tag}{savetag}.png'
    
    if mode=='auto':
        values, idxs, scores = utils.get_autoencoded(auto_fn, n_anoms=n_anoms, sigma=sigma)
    elif mode=='reals' or mode=='residuals':
        reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(results_fn, imarr_fn, n_anoms=n_anoms, sigma=sigma)
    
    if mode=='reals':
        values = reals    
    if mode=='residuals':
        residuals, _, _ = utils.get_residuals(reals, recons)
        values = residuals
   
    print(f"UMAP-ping {len(values)} values") 
    result = embed(values, idxs, scores, save_fn, n_neighbors=n_neighbors, min_dist=min_dist)
    if make_plot:
        plot(result, plot_fn)

    
def embed(values, idxs, colorby, save_fn, n_neighbors=5, min_dist=0.05):
    print("Reshaping") 
    values = np.array(values)
    values = values.reshape((values.shape[0], -1))
    print(values.shape)
    print("Embedding")
    #values = np.array(values).flatten() 
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist)
    embedding = reducer.fit_transform(values)

    result = np.array([embedding[:,0], embedding[:,1], colorby, idxs])
    print(save_fn)
    np.save(save_fn, result)
    
    return result

def plot_umap(result, plot_fn):
    e1, e2, colorby, idxs = result
    print("Plotting")
    plt.scatter(e1, e2, marker='.', c=colorby, cmap='viridis', s=8,
                                                                vmin=min(colorby), vmax=4000)
    plt.xlabel('umap 1')
    plt.ylabel('umap 2')
    
    cbar = plt.colorbar(extend='max')
    cbar.set_label('anomaly score', rotation=270, labelpad=10)
    #plt.clim(0,4000)
    plt.gca().set_aspect('equal', 'datalim')
    plt.savefig(plot_fn)
    

if __name__=='__main__':
    main()
