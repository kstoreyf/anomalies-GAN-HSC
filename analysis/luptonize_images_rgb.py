# ******************************************************
# * File Name : luptonize_reals_rgb.py
# * Creation Date : 2020-07-17
# * Created By : kstoreyf
# * Description : Luptonizes the real images and saves
#                 them to the results object
# ******************************************************

import numpy as np
import h5py

import utils

NSIDE = 96
NBANDS = 3

def main():
   
    #tag = 'gri_3sig'
    tag = 'gri_100k'
    #tag = 'gri_cosmos'

    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading data & residuals")
    imarr = h5py.File(imarr_fn, "r")
    res = h5py.File(results_fn, "a")
    reals = imarr['images']

    print("Luptonizing")
    reals = np.array(reals)
    reals = reals.reshape((-1,NSIDE,NSIDE,NBANDS))
    reals = utils.luptonize(reals).astype('int')

    print("Creating new dataset")
    res.create_dataset("reals", data=reals, dtype='uint8')

    imarr.close()
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
