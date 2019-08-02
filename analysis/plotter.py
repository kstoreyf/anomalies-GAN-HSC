# ********************************************************
# * File Name : plotter.py
# * Creation Date : 2019-07-30
# * Created By : kstoreyf
# * Description : Plotting routines:
#       - plot_comparisons to plot pairs of generated
#               images and their originals
#       - plot_dist to plot distribution of anomaly scores
# ********************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    #tag = 'i85k_96x96_norm'
    #tag = 'i20.0_norm_multigpu-0'
    tag = 'i20.0_norm_1layer'
    multi = True
    results_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/results'
    save_fn = f'{results_dir}/results_{tag}.png'
     
    if multi:
        fns = [f'{results_dir}/{fn}' for fn in os.listdir(results_dir) if tag in fn and fn.endswith('.npy')]
        res = np.concatenate([np.load(fn, allow_pickle=True)for fn in fns])
    else:
        results_fn = f'{results_dir}/results_{tag}.npy'
        res = np.load(results_fn, allow_pickle=True)
 
    print(f'Num results: {len(res)}')
    plot_dist(res, save_fn)
    
    
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
        step = int(len(scores)/(nrows*ncols))
        res = res[::step]
        
    if which=='anomalous':
        res = res[-(nrows*ncols):]

    reals = res[:,0]
    recons = res[:,1]
    resids = res[:,2]
    feat_resids = res[:,3] 
    scores = res[:,4]
    idxs = res[:,5]
    
    fig, axarr = plt.subplots(nrows,ncols, figsize=(15,15))
    #plt.subplots_adjust(hspace=0.4, wspace=0.2)
    cc = 0
    for i in range(nrows):
        for j in range(ncols):
            real = reals[cc].reshape((96,96))
            recon = recons[cc].reshape((96,96))
            combined = np.vstack((recon, real))
            axarr[i][j].imshow(combined, cmap='gray', origin='lower', interpolation='none')
            axarr[i][j].set_xticks([])
            axarr[i][j].set_yticks([])
            axarr[i][j].set_title('{}: {:.1f}\n ({:.1f}/{:.1f})'.format(idxs[cc], scores[cc], resids[cc], feat_resids[cc]), fontsize=10)
            cc += 1 

    plt.tight_layout() 
    plt.savefig(save_fn)#, bbox_inches='tight')


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
