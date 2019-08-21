# ********************************************************
# * File Name : plotter.py
# * Creation Date : 2019-07-30
# * Created By : kstoreyf
# * Description : Plotting routines:
#       - plot_comparisons to plot pairs of generated
#               images and their originals
#       - plot_dist to plot distribution of anomaly scores
#       - plot_anoms: plot batch of top anomalous images
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
import h5py

import save_images as saver
import utils


NBANDS=3

def main():
    
    tag = 'gri'
    plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-14'
    
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

    #savetag = '_genscore'
    savetag = ''
    #save_fn = f'{results_dir}/results_{tag}.png'
    comp_fn = f'{plot_dir}/comp_{tag}{savetag}.png'
    dist_fn = f'{plot_dir}/dist_{tag}{savetag}.png'
    anom_fn = f'{plot_dir}/anoms_{tag}{savetag}.png'

    res = h5py.File(results_fn)
    imarr = h5py.File(imarr_fn)
    #print(f'Num results: {len(res)}')
    #plot_dist(res[:,4], save_fn)
    #plot_comparisons(res, comp_fn, which='anomalous')
    #plot_comparisons(res, comp_fn, which='step')
    plot_comparisons(res, imarr, comp_fn, which='anomalous')
    plot_comparisons(res, imarr, comp_fn, which='step')
    plot_comparisons(res, imarr, comp_fn, which='random')
    plot_dist(res['anomaly_scores'], dist_fn)
    #plot_anoms(res, imarr, anom_fn)


def plot_anoms(res, imarr, anom_fn, n=128):
    idx_sorted = np.argsort(res['anomaly_scores'])
    sample = idx_sorted[-n:]
    reals = np.array([imarr['images'][s] for s in sample])
    #res = sort_by_score(res)
    reals = reals.reshape((-1,96,96,NBANDS))
    print(reals.shape)
    reals = utils.luptonize(reals).astype('int')
    #reals = np.stack(reals, axis=0)
    #print(reals)
    #print(reals.shape)
    saver.save_images(reals, anom_fn)
    
 
def sort_by_score(res):
    scores = res['anomaly_scores']
    idx = np.argsort(scores)
    print(idx)
    return res[list(idx)]


def plot_comparisons(res, imarr, save_fn, which='anomalous', sortby='anomaly_scores'):
    # TODO: make it plot to correct number of pixels!
    nrows = 3
    ncols = 8

    #res = sort_by_score(res)
    idx_sorted = np.argsort(res[sortby])

    if which=='step':
        step = int(len(res['idxs'])/(nrows*ncols))
        sample = idx_sorted[::step]
        
    if which=='anomalous':
        sample = list(idx_sorted[-(nrows*ncols):])

    if which=='random':
       sample = [int(r) for r in np.random.choice(len(res['idxs']), \
                size=nrows*ncols, replace=False)]

    if which=='check':
        sample = range(nrows*ncols)

    print(sample)
    
    reals = [imarr['images'][s] for s in sample]
    recons = [res['reconstructed'][s] for s in sample]
    resids = [res['gen_scores'][s] for s in sample]
    feat_resids = [res['disc_scores'][s] for s in sample]
    scores = [res['anomaly_scores'][s] for s in sample]
    idxs = [res['idxs'][s] for s in sample]
    
    labels = [res[sortby][s] for s in sample]

    fig, axarr = plt.subplots(nrows,ncols, figsize=(12,15))
    plt.subplots_adjust(left=0, right=1, top=0.96, bottom=0, hspace=0.1, wspace=0)
    cc = 0
    for i in range(nrows):
        for j in range(ncols):
            real = reals[cc].reshape((96,96,NBANDS))
            real = utils.luptonize(real).astype('int')
            recon = recons[cc].reshape((96,96,NBANDS)).astype('int')
            resid = abs(real-recon)

            if i==0 and j==0:
                print('real')
                print(real[40:-40,40:-40])
                print('recon')
                print(recon[40:-40,40:-40])
                print('resid')
                print(resid[40:-40,40:-40])
            combined = np.vstack((resid, recon, real))
            axarr[i][j].imshow(combined, cmap='gray', origin='lower', interpolation='none', aspect='equal')
            axarr[i][j].set_xticks([])
            axarr[i][j].set_yticks([])
            #axarr[i][j].set_title('{}: {:.1f}\n ({:.1f}/{:.1f})'.format(idxs[cc], scores[cc], resids[cc], feat_resids[cc]), fontsize=10)
            axarr[i][j].set_title('{:.2f}'.format(labels[cc]))
            axarr[i][j].axis('off')
            cc += 1 

    #plt.tight_layout() 
    plt.savefig(f'{save_fn[:-4]}_{which}.png', dpi=(96*ncols/12))#, bbox_inches='tight')


def plot_dist(scores_all, save_fn, labels=None):
    scores_all = np.array(scores_all)
    print(scores_all.shape)
    if not isinstance(scores_all[0], list) or isinstance(scores_all[0], np.ndarray):
        print("Single array, adding outer list")
        scores_all = [scores_all]
        colors = ['blue']
        lcolors = ['black']
    else:
        color_idx = np.linspace(0, 1, len(scores_all))
        colors = [plt.cm.rainbow(color_idx[i]) for i in range(len(scores_all))]
        lcolors = colors
    plt.figure()
    for i in range(len(scores_all)):
        scores = scores_all[i]
        color = colors[i]
        lcolor = lcolors[i]
        print(len(scores))
        mean = np.mean(scores)
        std = np.std(scores)
        print(len([s for s in scores if s>mean+3*std]))
        #color = plt.cm.rainbow(color_idx[i])
        if labels is not None:
            plt.hist(scores, bins=150, histtype='step', color=color, lw=2, label=labels[i])
            plt.legend()
        else:
            plt.hist(scores, bins=150, histtype='step', color=color, lw=2)
        plt.axvline(mean, lw=1, color=lcolor)
        plt.axvline(mean+std, lw=0.8, color=lcolor, ls='--')
        plt.axvline(mean-std, lw=0.8, color=lcolor, ls='--')
        plt.axvline(mean+2*std, lw=0.6, color=lcolor, ls='--')
        plt.axvline(mean-2*std, lw=0.6, color=lcolor, ls='--')
        plt.axvline(mean+3*std, lw=0.4, color=lcolor, ls='--')
        plt.axvline(mean-3*std, lw=0.4, color=lcolor, ls='--')
    plt.xlabel('anomaly score')
    plt.ylabel('#')
    #plt.xlim(500,3000)
    plt.savefig(save_fn)

if __name__=='__main__':
    main()
