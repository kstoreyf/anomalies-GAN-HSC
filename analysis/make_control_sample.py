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
import time


def main():
  
    imtag_orig = 'gri'
    restag_orig = 'gri_lambda0.3'
    imtag_new = 'gri_lambda0.3_control'
    restag_new = 'gri_lambda0.3_control' 

    start = time.time()

    # WARNING! this requires the imtag_orig and restag_orig to have the same objects in the same order. which it usually will, but should check 
    res_dir = '/scratch/ksf293/anomalies/results'
    res_fn_orig = f'{res_dir}/results_{restag_orig}.h5'
    res_fn_new = f'{res_dir}/results_{restag_new}.h5'
    data_dir = '/scratch/ksf293/anomalies/data'
    imarr_fn_orig = f'{data_dir}/images_h5/images_{imtag_orig}.h5'
    imarr_fn_new = f'{data_dir}/images_h5/images_{imtag_new}.h5'
    
    print("Loading results")
    res_orig = h5py.File(res_fn_orig, "r")
    print("Get sigma anomaly scores")
    gen_scores_sigma = res_orig['gen_scores_sigma'][:]
    disc_scores_sigma = res_orig['disc_scores_sigma'][:]
    res_orig.close()

    # choose opposite of 1.5sigdisc sample, and both scores above -1sigma
    # randomly choose 20k of these
    N = 20000
    locs_new = np.where((disc_scores_sigma > -1) & (gen_scores_sigma > -1) \
                     & ((disc_scores_sigma < gen_scores_sigma) | (disc_scores_sigma < 1.5)))[0]
    np.random.seed(42)
    locs_new = np.random.choice(locs_new, N, replace=False)
    print("N_sample:", len(locs_new))

    subsample_file(imarr_fn_orig, imarr_fn_new, locs_new)
    subsample_file(res_fn_orig, res_fn_new, locs_new)

    end = time.time()
    print("Time:", (end-start)/60.0, "min")

    print("Done!")


def subsample_file(file_fn_orig, file_fn_new, locs_new):
    print("Subsampling", file_fn_orig)
    file_orig = h5py.File(file_fn_orig, 'r')
    file_new = h5py.File(file_fn_new,"w")
    for key in file_orig.keys():
        print(key)
        arr_new = [file_orig[key][i] for i in locs_new]
        file_new.create_dataset(key, data=arr_new)
    file_orig.close()
    file_new.close()


if __name__=='__main__':
    main()
