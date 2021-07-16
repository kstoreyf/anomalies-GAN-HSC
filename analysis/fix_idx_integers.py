# ******************************************************
# * File Name : add_pdr2data_to_results.py
# * Creation Date :
# * Created By : kstoreyf
# * Description : Adds metadata from the HSC PDR2 catalog
#                 to the image datasets
# ******************************************************

import numpy as np
import pandas as pd
import h5py


def main():
   
    tag = 'gri_lambda0.3'
    aenum = 30000
    mode = 'reals'
    aetag = f'_latent64_{mode}_long_lr1e-4'
    savetag = f'_model{aenum}{aetag}'

    results_dir = f'/scratch/ksf293/anomalies/results'
    results_ae_fn = f'{results_dir}/results_ae_{tag}{savetag}.h5'

    print("Loading results")
    res = h5py.File(results_ae_fn, "a")
    idxs = [idx.astype(np.uint32) for idx in res['idxs'][:]]
    
    print("While we're at it, fix idxs -> integer")
    del res['idxs']
    res.create_dataset("idxs", data=idxs, dtype='uint32')

    res.close() 
    print("Done")


if __name__=='__main__':
    main()
