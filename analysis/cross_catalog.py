# **************************************************
# * File Name : cross_catalog.py
# * Creation Date : 2019-08-14
# * Created By : kstoreyf
# * Description :
# **************************************************

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import utils


plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-14b'

def main():
    cat_fn = "/scratch/ksf293/kavli/anomaly/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv"
    #cat_fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5.csv"
    cat = pd.read_csv(cat_fn)
    tag = 'gri'
    results_dir = '/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'
    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
    
    #savetag = '_log'
    savetag = '_alpha1'
    #savetag = ''

    sigma = 0
    reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(results_fn, imarr_fn, sigma=sigma)

    flag = 'blendedness_abs_flux'
    #flag = 'cmodel_ellipse_radius'
    #plot_vs_scores(idxs, scores, flag, cat, tag, savetag, band='i')

    flags = ['pixelflags_interpolated', 'pixelflags_saturated', 'pixelflags_clipped', 'pixelflags_rejected', 'pixelflags_inexact_psf', 'pixelflags_cr', 'pixelflags_clippedcenter', 'pixelflags_rejectedcenter', 'pixelflags_inexact_psfcenter']

    # get catalog with all of the flags = 1
    #flag = 'pixelflags_rejectedcenter'
    #flag = 'pixelflags_edge'
    calc_stats(idxs, scores, flag, cat)


def calc_stats(idxs, scores, flag, cat, band='i'):
    bflag = f'{band}_{flag}'
    values = cat.loc[idxs,bflag]
    mean = np.mean(scores)
    std = np.std(scores)

    thresh = 10**(-0.375)
    n_bad = len([v for v in values if v>thresh])
    print(f"Total number of objects: {len(values)}")
    p_bad = float(n_bad)/float(len(values))
    print(f"Number of blendedness values above threshhold: {n_bad}, {p_bad}%")


    arr_3sig = np.array([(s,v) for s,v in zip(scores, values) if s>mean+3*std])
    scores_3sig = arr_3sig[:,0]
    values_3sig = arr_3sig[:,1]
    n_3sig = len(arr_3sig)
    print(f"Number of 3 sigma anomalies: {n_3sig}")
    n_bad_3sig = len([v for v in values_3sig if v>thresh])
    p_bad_3sig = float(n_bad_3sig)/float(len(values_3sig))
    print(f"Number of blendedness values above threshhold pf 3sigma anomalies: {n_bad_3sig}, {p_bad_3sig}%")

    
def plot_vs_scores(idxs, scores, flag, cat, tag, savetag, band='i'):

    bflag = f'{band}_{flag}'
    mean = np.mean(scores)
    std = np.std(scores)

    print(len([s for s in scores if s>4000]))
    print("get iblend")
    values = cat.loc[idxs,bflag]
    print("plot")
        
    plt.scatter(scores, values, marker='.', s=8, alpha=1)

    lcolor='k'
    plt.axvline(mean, lw=1, color=lcolor)
    plt.axvline(mean+std, lw=0.8, color=lcolor, ls='--')
    plt.axvline(mean-std, lw=0.8, color=lcolor, ls='--')
    plt.axvline(mean+2*std, lw=0.6, color=lcolor, ls='--')
    plt.axvline(mean-2*std, lw=0.6, color=lcolor, ls='--')
    plt.axvline(mean+3*std, lw=0.4, color=lcolor, ls='--')
    plt.axvline(mean-3*std, lw=0.4, color=lcolor, ls='--')    

    plt.xlabel("anomaly score")
    plt.ylabel(bflag)
    plt.xlim(min(scores), 4000)
    #plt.ylim(-1, 11)
    if flag=='blendedness_abs_flux':
        plt.ylim(-0.1, 1.1)
        thresh = 10**(-0.375)
        plt.axhline(thresh, color='red')
    plt.savefig(f"{plot_dir}/scores_{bflag}_{tag}{savetag}.png")
    
    #print(cat_flagged.loc[:,[f'{band}_{flag}' for band in bands]])
    #nflags = cat_flagged.loc[:,[f'{band}_{flag}' for band in bands]].sum(axis=1)
    #nflags = cat.loc[:,[f'{band}_{flag}' for band in bands]].sum(axis=1)
    #catone = cat[nflags>0]
    #print(len(catone))
    #print(nflags)
    #print(catone.loc[:,[f'{band}_{flag}' for band in bands]])
    #for b in range(len(bands)):
    #    band = bands[b]
    #    print(cat_flagged[f'{band}_{flag}'])
    #count = 0

def count_flags(idxs, flag, bands):
    count = 0
    for i in range(len(idxs)):
        idx = int(idxs[i])
        countflag = 0
        for b in range(len(bands)):
            cflag = '{}_{}'.format(bands[b], flag)
            if cat.iloc[idx][cflag]>thresh:
                countflag += 1
        if countflag > 0:
            count += 1
    return count

    #cat_flagged = cat
    #for b in range(len(bands)):
    #    cflag = '{}_{}'.format(bands[b], flag)
    #    cat_flagged = cat_flagged[cat_flagged[cflag]>thresh]
    #print(len(cat_flagged))
    #print(float(len(cat_flagged))/len(cat))
        
        #if object_ids[i] in cat_flagged['object_id']:
        #    count += 1
    #print(count)
    #print(len(idxs))
    #print(float(count)/float(len(idxs)))








if __name__=='__main__':
    main()

