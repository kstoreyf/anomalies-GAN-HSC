# ******************************************************
# * File Name : add_sigma_scores.py
# * Creation Date : 
# * Created By : kstoreyf
# * Description : Add the normalized scores (sigma,  
#                 standard deviation) to the datasets
# ******************************************************

import numpy as np
import pandas as pd
import h5py


def main():
   
    #tag = 'gri_1k_lambda0.3'
    tag = 'gri_lambda0.3'

    aenum = 30000
    mode = 'reals'
    aetag = f'_latent64_{mode}_long_lr1e-4'
    savetag = f'_model{aenum}{aetag}'

    results_dir = f'/scratch/ksf293/anomalies/results'
    results_ae_fn = f'{results_dir}/results_aefull_{tag}{savetag}.h5'
    print(results_ae_fn)

    print("Loading results")
    res = h5py.File(results_ae_fn, "a")
    print(res.keys())
    
    print("Computing sigma anomaly scores")
    ae_scores = res['ae_anomaly_scores'][:]

    ae_scores_sigma = (ae_scores - np.mean(ae_scores))/np.std(ae_scores) 

    #print("Cleanup existing datasets")
    #new_keys = ["gen_scores_norm", "disc_scores_norm", "anomaly_scores_norm",
    #            "gen_scores_rank", "disc_scores_rank", "anomaly_scores_rank"]
    #for nk in new_keys:
    #    if nk in res.keys():
    #        del res[nk]

    print("Creating new datasets for sigma scores")
    res.create_dataset("ae_anomaly_scores_sigma", data=ae_scores_sigma)
    
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
