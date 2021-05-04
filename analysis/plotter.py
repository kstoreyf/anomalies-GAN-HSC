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
import pandas as pd
import h5py

import save_images as saver
import utils


base_dir = '/scratch/ksf293/anomalies'
NBANDS=3

def main():
    
    tag = 'gri'
    plot_dir = f'/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-14b'
    
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

    #savetag = '_genscore'
    savetag = '_max4000'
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
    #plot_comparisons(res, imarr, comp_fn, which='step')
    #plot_comparisons(res, imarr, comp_fn, which='random')
    #plot_dist(res['anomaly_scores'], dist_fn)
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


def plot_comparisons(res, imarr, save_fn, which='anomalous', sortby='anomaly_scores', sample=[], luptonize=True):
    # TODO: make it plot to correct number of pixels!
    nrows = 3
    ncols = 8

    #res = sort_by_score(res)
    idx_sorted = np.argsort(res[sortby])

    if which=='sample':
        sample = sample

    if which=='step':
        step = int(len(res['idxs'])/(nrows*ncols))
        sample = idx_sorted[::step]
        
    if which=='anomalous':
        sample = list(idx_sorted[-(nrows*ncols)-84:-84]) #remove >4000 anomalies for all sample
        #sample = list(idx_sorted[-(nrows*ncols):])

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
            if luptonize:
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
    plt.xlim(0,4000)
    plt.savefig(save_fn)

def plot_ims_tight(ids, nrows, ncols, saveto=None):
    NSIDE = 96
    NBANDS = 3
    imgrid = np.empty((nrows*NSIDE, ncols*NSIDE, NBANDS))

    imdict_fn = f'{base_dir}/data/idxdicts_h5/idx2imloc_gri.npy'
    idx2imloc = np.load(imdict_fn, allow_pickle=True).item()
    imarr_fn = f'{base_dir}/data/images_h5/images_gri.h5'
    imarr = h5py.File(imarr_fn, 'r')

    dpi = 96
    plt.figure(figsize=(NSIDE*ncols/dpi, NSIDE*nrows/dpi), dpi=dpi)
    #fig = plt.figure()
    #fig.set_size_inches(NSIDE*ncols/dpi, NSIDE*nrows/dpi)
    ax = plt.gca()
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            idx = ids[count]
            loc = idx2imloc[idx]
            im = imarr['images'][loc]
            im = utils.luptonize(im)
            
            imgrid[i*NSIDE:(i+1)*NSIDE, j*NSIDE:(j+1)*NSIDE, :] = im
            count += 1
    imgrid = imgrid.astype('int32')
    ax.imshow(imgrid)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis("off")
    if saveto:
        #plt.savefig(saveto, bbox_inches='tight', dpi=dpi*1.27)
        plt.savefig(saveto, bbox_inches='tight', dpi=dpi)
        #plt.savefig(saveto, dpi=dpi)

def plot_ims(ids, nrows, ncols, imtag='gri', saveto=None, headers=None, 
             subsize=2, tight=False, hspace=None, wspace=None, 
             border_indices=None, border_color='cyan', restag=None,
             resdicttag=None, score_name=None, score_label=None,
             **kwargs):
    assert len(ids)<=nrows*ncols, "bad rows/cols for number of ids!"

    if resdicttag is None:
        resdicttag = restag

    if score_name is not None:
        resdict_fn = f'{base_dir}/data/idxdicts_h5/idx2resloc_{resdicttag}.npy' # THIS MIGHT NEED TO BE RESTAG
        idx2resloc = np.load(resdict_fn, allow_pickle=True).item()
        results_fn = f'{base_dir}/results/results_{restag}.h5'
        res = h5py.File(results_fn, 'r')

    imdict_fn = f'{base_dir}/data/idxdicts_h5/idx2imloc_{imtag}.npy'
    idx2imloc = np.load(imdict_fn, allow_pickle=True).item()
    imarr_fn = f'{base_dir}/data/images_h5/images_{imtag}.h5'
    imarr = h5py.File(imarr_fn, 'r')
 
    fig, axarr = plt.subplots(nrows,ncols,figsize=(ncols*subsize,nrows*subsize))
    if hspace is None and tight:
        hspace = 0.03
        wspace = -0.15
    elif hspace is None and score_name == None:
        hspace=0.2
        wspace=0.05
    print(hspace,wspace)
    plt.subplots_adjust(hspace=hspace, wspace=wspace)
    plt.rc('text', usetex=True)
    
    count = 0
    for i in range(nrows):
        for j in range(ncols):
            idx = ids[count]
            loc = idx2imloc[idx]
            im = imarr['images'][loc]
            obj_id = int(imarr['object_ids'][loc])
            if score_name is not None:
                rloc = idx2resloc[idx]
                score = float(res[score_name][rloc])
                sdisc = float(res['disc_scores_sigma'][rloc])
                sgen = float(res['gen_scores_sigma'][rloc])
            if nrows==1 and ncols==1:
                ax = axarr
            elif nrows==1:
                ax = axarr[j]
            elif ncols==1:
                ax = axarr[i]
            else:
                ax = axarr[i][j]
                
            if score_name is not None and not tight:
                #title = f"ID: {obj_id}"
                title = r'''ID: {}
{} = {:.2f}$\sigma$
{:.2f}, {:.2f}'''.format(obj_id, score_label, score, sdisc, sgen)
                ax.set_title(title, fontsize=8)
            
            ax.imshow(utils.luptonize(im, **kwargs))
            ax.set_xticks([])
            ax.set_yticks([])
            
            if headers is not None and i==0:
                # units are pixel values;
                nside = 96
                ax.text(nside/2,-nside/4,headers[j], size=16, horizontalalignment='center')
                
            count += 1
            if count>=len(ids):
                break
                
        if count>=len(ids):
            break
    
    if border_indices is not None:
        bi, bj = border_indices
        axarr[bi][bj].patch.set_edgecolor(border_color)
        axarr[bi][bj].patch.set_linewidth('6')
                    
    if saveto:
        plt.savefig(saveto, bbox_inches='tight')#, pad_inches=0)


def plot_umap(embedding, saveto=None, highlight_arrs=None, highlight_colors=None,
              highlight_markers=None, cmap='plasma_r', boxes=None, box_colors=None,
              box_labels=None, figsize=(8,7), colorby=None, vmin=None, vmax=None, 
              alpha=0.2, s=6, xlim=None, ylim=None, show_axes=False):
    e1, e2, cby, idxs = embedding
    if colorby is None:
        colorby = cby

    if vmin is None:
        vmin = min(colorby)
    if vmax is None:
        vmax = 0.35*max(colorby)

    plt.figure(figsize=figsize)
    plt.scatter(e1, e2, marker='.', c=colorby, cmap=cmap, s=s, vmin=vmin, vmax=vmax, alpha=alpha)

    if highlight_arrs is not None:
        if np.array(highlight_arrs).ndim==1:
            highlight_arrs = [highlight_arrs]
        for i, highlight_ids in enumerate(highlight_arrs):
            argidxs = [np.where(idxs==hi)[0][0] for hi in highlight_ids]
            plt.scatter(e1[argidxs], e2[argidxs], marker=highlight_markers[i], c=colorby[argidxs],
                            edgecolor=highlight_colors[i], lw=2,
                            cmap=cmap, s=60, vmin=vmin, vmax=vmax)

    if boxes is not None:
        for i, box in enumerate(boxes):
            amin, amax, bmin, bmax = boxes[i]
            width = amax - amin
            height = bmax - bmin
            rect = matplotlib.patches.Rectangle((amin,bmin),width,height,linewidth=3,
                                                edgecolor=box_colors[i],facecolor='none')
            ax = plt.gca()
            ax.add_patch(rect)
            ax.text(amin-0.8, bmin, box_labels[i], fontsize=18)

    if show_axes:
        plt.xlabel('umap A')
        plt.ylabel('umap B')
    else:
        ax = plt.gca()
        ax.set_xticks([])
        ax.set_yticks([])
        #ax.axis("off")

    plt.xlim(xlim)
    plt.ylim(ylim)

    cbar = plt.colorbar(extend='max')
    cbar.set_label(r'$s_\mathrm{disc}$, discriminator score', rotation=270, labelpad=18)
    cbar.set_alpha(1)
    cbar.draw_all()

    plt.gca().set_aspect('equal', 'datalim')

    if saveto:
        plt.savefig(saveto, bbox_inches='tight')

    return plt.gca()


NSIDE = 96
def get_residual(im, recon):
    im = np.array(im)
    reals = im.reshape((NSIDE,NSIDE,-1))
    recon = np.array(recon)
    recon = recon.reshape((NSIDE,NSIDE,-1)).astype('int')
    resid = abs(im-recon)
    return resid


def plot_recons(ids, imtag, restag, resdicttag=None, saveto=None, border_color=None,
                score_name='disc_scores_sigma', score_label='$s_\mathrm{{disc}}$'):
    nims = len(ids)
    if resdicttag is None:
        resdicttag = restag
    imdict_fn = f'{base_dir}/data/idxdicts_h5/idx2imloc_{imtag}.npy'
    resdict_fn = f'{base_dir}/data/idxdicts_h5/idx2resloc_{resdicttag}.npy' # THIS MIGHT NEED TO BE RESTAG
    idx2imloc = np.load(imdict_fn, allow_pickle=True).item()
    idx2resloc = np.load(resdict_fn, allow_pickle=True).item()

    imarr_fn = f'{base_dir}/data/images_h5/images_{imtag}.h5'
    imarr = h5py.File(imarr_fn, 'r')
    results_fn = f'{base_dir}/results/results_{restag}.h5'
    res = h5py.File(results_fn, 'r')
    subsize = 2
    fig, axarr = plt.subplots(3,nims,figsize=(nims*subsize,3*subsize), edgecolor=border_color)
    plt.subplots_adjust(hspace=0.02, wspace=0.1)
    count = 0
    for i in range(nims):
        idx = ids[i]
        
        loc = idx2imloc[idx]
        im = utils.luptonize(imarr['images'][loc])
        
        rloc = idx2resloc[idx]
        recon = res['reconstructed'][rloc]
        score = float(res[score_name][rloc])
        obj_id = int(res['object_ids'][rloc])
        if 'residual' in res.keys():
            resid = res['residual'][rloc]
        else:
            resid = get_residual(im, recon)
        
        ax0 = axarr[0][i]
        ax1 = axarr[1][i]
        ax2 = axarr[2][i]
        title = r'''ID: {}
{} = {:.2f}$\sigma$'''.format(obj_id, score_label, score)
        ax0.set_title(title, fontsize=8)
        ax0.imshow(im)
        ax1.imshow(recon)
        ax2.imshow(resid)
        
        if i==0:
            fsize=13
            ax0.set_ylabel("real",fontsize=fsize)
            ax1.set_ylabel("reconstructed",fontsize=fsize)
            ax2.set_ylabel("residual",fontsize=fsize)
        
        for ax in [ax0, ax1, ax2]:
            ax.set_xticks([])
            ax.set_yticks([])
    
    if border_color is not None:
        fig.patch.set_edgecolor(border_color)  
        fig.patch.set_linewidth('4') 
        
    if saveto:
        plt.savefig(saveto, bbox_inches='tight', edgecolor=border_color)#, pad_inches=0)


def plot_anomaly_dist(sanoms, gens, discs, title=None, saveto=None):
    
    print(min(gens), max(gens), np.mean(gens), np.std(gens))
    print(min(discs), max(discs), np.mean(discs), np.std(discs))

    minmin = min(min(gens), min(discs))
    maxmax = max(max(gens), max(discs))
    bins = np.linspace(minmin, maxmax, 200)
    
    fig, axarr = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={'width_ratios': [1, 1.5]})
    plt.subplots_adjust(wspace=0)
    
    fig.suptitle(title)

    ax0 = axarr[0]
    b = ax0.hist(sanoms, bins=bins, alpha=1, color='purple', label='total \nscore $(s_\mathrm{anom})$', histtype='step', lw=1.5)
    b = ax0.hist(gens, bins=bins, alpha=1, color='blue', ls=':', label='generator \nscore $(s_\mathrm{gen})$', histtype='step', lw=1.5)
    b = ax0.hist(discs, bins=bins, alpha=1, color='red', ls='--', label='discriminator \nscore $(s_\mathrm{disc})$', histtype='step', lw=1.5)

    mean = np.mean(sanoms)
    std = np.std(sanoms)
    thresh_3sig = mean+3*std
    
    plot_sigma_lines = False
    if plot_sigma_lines:
    	lcolor='k'
    	ax0.axvline(mean, lw=1, color=lcolor)
    	ax0.axvline(mean+std, lw=0.8, color=lcolor, ls='--')
    	ax0.axvline(mean-std, lw=0.8, color=lcolor, ls='--')
    	ax0.axvline(mean+2*std, lw=0.6, color=lcolor, ls='--')
    	ax0.axvline(mean-2*std, lw=0.6, color=lcolor, ls='--')
    	ax0.axvline(mean+3*std, lw=0.4, color=lcolor, ls='--')
    	ax0.axvline(mean-3*std, lw=0.4, color=lcolor, ls='--')
    
    ax0.legend()
    ax0.set_xlabel(r"score ($\sigma$)")
    ax0.set_ylabel("number")
    ax0.set_xlim(-2, 5)
    
    
    ax1 = axarr[1]
    scat = ax1.scatter(gens, discs, s=1, c=sanoms, alpha=0.2, cmap='plasma_r', vmin=-2, vmax=5)
    cbar = fig.colorbar(scat, extend='max', ax=ax1)
    cbar.set_label(r'$s_\mathrm{anom}$ ($\sigma$)', rotation=270, labelpad=8)
    cbar.set_alpha(1)
    cbar.draw_all()
   
    smin, smax = -5, 15 
    xx = np.linspace(smin,smax,2)
    ax1.plot(xx, xx, color='k', ls='--', lw=0.5)
    #lambda_weight = 0.5
    #line_3sig = thresh_3sig/lambda_weight - xx
    #plt.plot(xx, line_3sig, lw=0.4, color=lcolor, ls='--')

    ax1.set_xlabel(r"$s_\mathrm{gen}$, generator score ($\sigma$)")
    ax1.set_ylabel(r"$s_\mathrm{disc}$, discriminator score ($\sigma$)")

    ax1.set_xlim(smin,smax)
    ax1.set_ylim(smin, smax)	
    ticks = np.arange(smin, smax+5, 5)
    ax1.set_xticks(ticks)
    ax1.set_yticks(ticks)
    ax1.set_aspect('equal', adjustable='box')

    plt.show()    
    if saveto:
        plt.savefig(saveto, bbox_inches='tight')


def plot_cross_hexbin(results_fn, cat_fn, x_name, cmap='coolwarm', saveto=None):
    print("Loading data")
    res = h5py.File(results_fn, 'r')
    scores = res['disc_scores_sigma']
    idxs = [int(idx) for idx in res['idxs'][:]]

    print("Loading catalog")
    x_col_name = {'extendedness': 'i_cmodel_ellipse_radius',
                  'blendedness': 'i_blendedness_abs_flux'}
    cols = [x_col_name[x_name], 'idx']
    cat = pd.read_csv(cat_fn, usecols=cols, squeeze=True)

    print("Getting blend & extend")
    x_vals = [cat[x_col_name[x_name]].iloc[idx] for idx in idxs]
    scores = [s if s<=5 else 5 for s in scores] #pile extreme scores into edge bins
    scores = [s if s>=-2 else -2 for s in scores]

    # if need to save intermediate result
    #result = np.array([extend, blend, scores, idxs])
    #np.save(save_fn, result)

    print("Plotting")
    plt.figure(figsize=(10,8))
    #plt.scatter(extend, scores, marker='.', c=blend, cmap=cmap, s=16, alpha=0.3)
    #plt.contourf(extend, scores, blend, cmap=cmap)
    
    plt.ylabel(r'$s_\mathrm{disc}$, discriminator score')
    
    if x_name=='extendedness':
        #x_vals = [np.log10(x) for x in x_vals]
        #print(min(x_vals), max(x_vals))
        xscale = 'log'
        plt.xlabel(r'$R_\mathrm{eff}$, effective radius (arcsec)')
    if x_name=='blendedness':
        x_vals = [x if x>=0 else 0 for x in x_vals]
        xscale = 'linear'
        thresh = 10**(-0.375)
        plt.axvline(thresh, color='purple', ls='--', lw=1.5)
        plt.xlabel('blendedness')
    plt.hexbin(x_vals, scores, gridsize=30, cmap='Blues', bins='log', xscale=xscale)

    cbar = plt.colorbar()
    cbar.set_label('number in bin', rotation=270, labelpad=18)
    
    lcolor='k'
    plt.axhline(3, lw=0.75, color=lcolor)
   
    res.close()
    
    if saveto:
        plt.savefig(saveto, bbox_inches='tight', pad_inches=0.1)


if __name__=='__main__':
    main()
