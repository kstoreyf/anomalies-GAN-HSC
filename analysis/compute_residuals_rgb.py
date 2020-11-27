# ******************************************************
# * File Name : compute_residuals_rgb.py
# * Creation Date : 2020-07-17
# * Created By : kstoreyf
# * Description : Computes the image residuals from the
#                 real & reconstructeds, for RGB images
# ******************************************************

import numpy as np
import h5py

import utils


def main():
   
    #tag = 'gri'
    #tag = 'gri_3signorm'
    tag = 'gri_100k'
    #tag = 'gri_cosmos'

    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading data & residuals")
    imarr = h5py.File(imarr_fn, "r")
    res = h5py.File(results_fn, "a")
    reals = imarr['images']
    recons = res['reconstructed']

    resids = utils.get_residuals(reals, recons)

    print("Creating new dataset")
    if 'residuals' in res.keys():
        del res['residuals']
    res.create_dataset("residuals", data=resids, dtype='uint8')

    imarr.close()
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
