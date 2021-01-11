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
    n_new = 10000
    tag_new = 'gri_10k' 

    start = time.time()

    res_dir = '/scratch/ksf293/kavli/anomaly/results'
    res_fn_orig = f'{res_dir}/results_{tag_orig}.h5'
    res_fn_new = f'{res_dir}/results_{tag_new}.h5'
    data_dir = '/scratch/ksf293/kavli/anomaly/data'
    imarr_fn_orig = f'{data_dir}/images_h5/images_{tag_orig}.h5'
    imarr_fn_new = f'{data_dir}/images_h5/images_{tag_new}.h5'
    
    print("Loading results")
    imarr_orig = h5py.File(imarr_fn_orig, "r")
    print("Get normalized anomaly scores")
    idxs = imarr_orig['idxs'][:]
    locs = np.arange(len(idxs))
    imarr_orig.close()

    locs_new = np.random.choice(locs, size=n_new, replace=False)

    subsample_file(imarr_fn_orig, imarr_fn_new, locs_new)
    #subsample_file(res_fn_orig, res_fn_new, locs_3sig)

    end = time.time()
    print("Time:", (end-start)/60.0, 'min')

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
