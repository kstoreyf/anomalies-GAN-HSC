# ********************************************************
# * File Name : plotter.py
# * Creation Date : 2019-07-30
# * Created By : kstoreyf
# * Description : Plotting routines:
#       - plot_comparisons to plot pairs of generated
#               images and their originals
#       - plot_dist to plot distribution of anomaly scores
# ********************************************************
import os
import sys
from os import path
#sys.path.append( path.dirname( path.dirname( path.abspath(__file__) ) ) )
#from gans.tflib import save_images as saver

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

import save_images as saver
#from ..gans.tflib import save_images as saver

def main():
    #tag = 'i85k_96x96_norm'
    #tag = 'i20.0_norm_multigpu-0'
    #tag = 'i20.0_norm_batch_ano0.05'
    tag = 'i20.0_norm_features0.05go'
    #tag = 'i20.0_norm_features'
    #multi = True
    multi = False
    plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-01'
    #results_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/results'
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    #save_fn = f'{results_dir}/results_{tag}.png'
    comp_fn = f'{plot_dir}/comp_{tag}.png'
    dist_fn = f'{plot_dir}/dist_{tag}.png'
    anom_fn = f'{plot_dir}/anoms_{tag}.png'

    if multi:
        fns = [f'{results_dir}/{fn}' for fn in os.listdir(results_dir) if tag in fn and fn.endswith('.npy')]
        res = np.concatenate([np.load(fn, allow_pickle=True)for fn in fns])
    else:
        results_fn = f'{results_dir}/results_{tag}.npy'
        res = np.load(results_fn, allow_pickle=True)
 
    print(f'Num results: {len(res)}')
    #plot_dist(res, save_fn)
    plot_comparisons(res, comp_fn, which='anomalous')
    plot_comparisons(res, comp_fn, which='step')
    #plot_dist(res, dist_fn)
    #plot_anoms(res, anom_fn)


def plot_anoms(res, anom_fn, n=128):
    res = sort_by_score(res)
    res = res[-n:]
    reals = res[:,0]
    reals = np.stack(reals, axis=0)
    print(reals)
    print(reals.shape)
    saver.save_images(reals.reshape((n, 96, 96)), anom_fn)
    
 
def sort_by_score(res):
    scores = res[:,4]
    idx = np.argsort(scores)
    return res[idx]


def plot_comparisons(res, save_fn, which='anomalous'):
    # TODO: make it plot to correct number of pixels!
    nrows = 4
    ncols = 8

    res = sort_by_score(res)

    if which=='step':
        step = int(len(res)/(nrows*ncols))
        res = res[::step]
        
    if which=='anomalous':
        res = res[-(nrows*ncols):]

    reals = res[:,0]
    recons = res[:,1]
    resids = res[:,2]
    feat_resids = res[:,3] 
    scores = res[:,4]
    idxs = res[:,5]
    
    fig, axarr = plt.subplots(nrows,ncols, figsize=(12,16))
    plt.subplots_adjust(left=0, right=1, top=0.95, bottom=0, hspace=0.1, wspace=0)
    cc = 0
    for i in range(nrows):
        for j in range(ncols):
            real = reals[cc].reshape((96,96))
            recon = recons[cc].reshape((96,96))
            combined = np.vstack((recon, real))
            axarr[i][j].imshow(combined, cmap='gray', origin='lower', interpolation='none', aspect='equal')
            axarr[i][j].set_xticks([])
            axarr[i][j].set_yticks([])
            axarr[i][j].set_title('{}: {:.1f}\n ({:.1f}/{:.1f})'.format(idxs[cc], scores[cc], resids[cc], feat_resids[cc]), fontsize=10)
            axarr[i][j].axis('off')
            cc += 1 

    #plt.tight_layout() 
    plt.savefig(f'{save_fn[:-4]}_{which}.png', dpi=(96*ncols/12))#, bbox_inches='tight')


def plot_dist(res, save_fn):
    scores = res[:,4]
    mean = np.mean(scores)
    std = np.std(scores)
    print(len([s for s in scores if s>mean+3*std]))

    plt.figure()
    plt.hist(scores, bins=60, histtype='step', color='blue', lw=2)
    plt.axvline(mean, lw=3, color='k')
    plt.axvline(mean+std, lw=2, color='k', ls='--')
    plt.axvline(mean-std, lw=2, color='k', ls='--')
    plt.axvline(mean+2*std, lw=1, color='k', ls='--')
    plt.axvline(mean-2*std, lw=1, color='k', ls='--')
    plt.axvline(mean+3*std, lw=0.5, color='k', ls='--')
    plt.axvline(mean-3*std, lw=0.5, color='k', ls='--')
    plt.xlabel('anomaly score')
    plt.ylabel('#')
    plt.savefig(save_fn)

if __name__=='__main__':
    main()
