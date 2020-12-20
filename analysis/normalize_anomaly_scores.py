# ******************************************************
# * File Name : compute_residuals_rgb.py
# * Creation Date : 2020-07-17
# * Created By : kstoreyf
# * Description : Computes the image residuals from the
#                 real & reconstructeds, for RGB images
# ******************************************************

import numpy as np
import pandas as pd
import h5py


def main():
   
    #tag = 'gri_3sig'
    #tag = 'gri_100k'
    #tag = 'gri_cosmos'
    tag = 'gri'

    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading results")
    res = h5py.File(results_fn, "a")
    
    print("Computing normalized anomaly scores")
    gen_scores = res['gen_scores'][:]
    disc_scores = res['disc_scores'][:]

    gen_norm = (gen_scores - np.mean(gen_scores))/np.std(gen_scores)
    disc_norm = (disc_scores - np.mean(disc_scores))/np.std(disc_scores)
    
    minmin = min(min(gen_norm), min(disc_norm))
    maxmax = max(max(gen_norm), max(disc_norm))
    gen_norm = norm(gen_norm, minmin, maxmax)
    disc_norm = norm(disc_norm, minmin, maxmax)
    
    ag = 0.6
    ad = 0.15
    gen_norm = np.arcsinh(gen_norm/ag)*ag
    disc_norm = np.arcsinh(disc_norm/ad)*ad
    
    minmin = min(min(gen_norm), min(disc_norm))
    maxmax = max(max(gen_norm), max(disc_norm))
    gen_norm = norm(gen_norm, minmin, maxmax)
    disc_norm = norm(disc_norm, minmin, maxmax)

    lambda_weight = 0.5
    score_norm = (1-lambda_weight)*gen_norm + lambda_weight*disc_norm

    print("Cleanup existing datasets")
    new_keys = ["gen_scores_norm", "disc_scores_norm", "anomaly_scores_norm",
                "gen_scores_rank", "disc_scores_rank", "anomaly_scores_rank"]
    for nk in new_keys:
        if nk in res.keys():
            del res[nk]

    print("Creating new datasets for normalized")
    res.create_dataset("gen_scores_norm", data=gen_norm)
    res.create_dataset("disc_scores_norm", data=disc_norm)
    res.create_dataset("anomaly_scores_norm", data=score_norm)
    
    print("Computing score ranks")
    gen_order = np.argsort(gen_scores)
    disc_order = np.argsort(disc_scores)
    gen_rank = np.argsort(gen_order)
    disc_rank = np.argsort(disc_order)
    score_rank = np.sqrt(gen_rank*disc_rank)

    print("Creating new datasets for rank ordering")
    res.create_dataset("gen_scores_rank", data=gen_rank)
    res.create_dataset("disc_scores_rank", data=disc_rank)
    res.create_dataset("anomaly_scores_rank", data=score_rank)

    res.close() 
    print("Done")


def norm(a, nmin, nmax):
    return (a - nmin)/(nmax-nmin)   


if __name__=='__main__':
    main()
