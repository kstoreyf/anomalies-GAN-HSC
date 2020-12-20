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
  
    tag_orig = 'gri'
    tag_new = 'gri_3signorm' 

    start = time.time()

    res_dir = '/scratch/ksf293/kavli/anomaly/results'
    res_fn_orig = f'{res_dir}/results_{tag_orig}.h5'
    res_fn_new = f'{res_dir}/results_{tag_new}.h5'
    data_dir = '/scratch/ksf293/kavli/anomaly/data'
    imarr_fn_orig = f'{data_dir}/images_h5/images_{tag_orig}.h5'
    imarr_fn_new = f'{data_dir}/images_h5/images_{tag_new}.h5'
    
    print("Loading results")
    res_orig = h5py.File(res_fn_orig, "r")
    print("Get normalized anomaly scores")
    scores = res_orig['anomaly_scores_norm'][:]
    res_orig.close()

    thresh_3sig = np.mean(scores) + 3*np.std(scores)
    locs_3sig = np.where(scores>thresh_3sig)[0]

    #subsample_file(imarr_fn_orig, imarr_fn_new, locs_3sig)
    subsample_file(res_fn_orig, res_fn_new, locs_3sig)

    end = time.time()
    print("Time:", (end-start)/60.0)

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
