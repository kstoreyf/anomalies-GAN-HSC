# ******************************************************
# * File Name : luptonize_images_rgb.py
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
   
    #tag = 'gri'
    #tag = 'gri_3signorm'
    #tag = 'gri_100k'
    #tag = 'gri_lambda0.3_1.5sigdisc'
    imtag = 'gri_lambda0.3_3sigd'
    tag = 'gri_lambda0.3_3sigd'   

    imarr_fn = f'/scratch/ksf293/anomalies/data/images_h5/images_{imtag}.h5'
    results_dir = f'/scratch/ksf293/anomalies/results'
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
