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
   
    #restag = 'gri_10k_lambda0.3'
    restag = 'gri_100k_lambda0.3'
    #restag = 'gri'

    results_dir = f'/scratch/ksf293/anomalies/results'
    results_fn = f'{results_dir}/results_{restag}.h5'

    print("Loading results")
    res = h5py.File(results_fn, "a")
    
    print("Computing sigma anomaly scores")
    gen_scores = res['gen_scores'][:]
    disc_scores = res['disc_scores'][:]
    scores = res['anomaly_scores'][:]

    gen_scores_sigma = (gen_scores - np.mean(gen_scores))/np.std(gen_scores)
    disc_scores_sigma = (disc_scores - np.mean(disc_scores))/np.std(disc_scores)
    scores_sigma = (scores - np.mean(scores))/np.std(scores) 

    #print("Cleanup existing datasets")
    #new_keys = ["gen_scores_norm", "disc_scores_norm", "anomaly_scores_norm",
    #            "gen_scores_rank", "disc_scores_rank", "anomaly_scores_rank"]
    #for nk in new_keys:
    #    if nk in res.keys():
    #        del res[nk]

    print("Creating new datasets for sigma scores")
    res.create_dataset("gen_scores_sigma", data=gen_scores_sigma)
    res.create_dataset("disc_scores_sigma", data=disc_scores_sigma)
    res.create_dataset("anomaly_scores_sigma", data=scores_sigma)
    
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
