# ******************************************************
# * File Name : add_reals_to_results.py
# * Creation Date :
# * Created By : kstoreyf
# * Description : Adds luptonized real images to results
#                 datasets
# ******************************************************

import numpy as np
import h5py

import utils


def main():
   
    #tag = 'gri'
    #tag = 'gri_3signorm'
    #imtag = 'gri_lambda0.3_1.5sigdisc'
    #tag = 'gri_lambda0.3_1.5sigdisc'
    #imtag = 'gri_lambda0.3_control'
    #tag = 'gri_lambda0.3_control'
    imtag = 'gri_100k'
    tag = 'gri_100k_lambda0.3'

    imarr_fn = f'/scratch/ksf293/anomalies/data/images_h5/images_{imtag}.h5'
    results_dir = f'/scratch/ksf293/anomalies/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading data & residuals")
    imarr = h5py.File(imarr_fn, "r")
    res = h5py.File(results_fn, "a")
    reals = imarr['images'][:]

    NBANDS = 3
    reals = reals.reshape((-1,96,96,NBANDS))
    reals = utils.luptonize(reals).astype('int')
    imarr.close()

    print("Creating new dataset")
    if 'reals' in res.keys():
        del res['reals']
    res.create_dataset("reals", data=reals, dtype='uint8')

    res.close() 
    print("Done")


if __name__=='__main__':
    main()
