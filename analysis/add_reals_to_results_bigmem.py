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
    #imtag = 'gri_lambda0.3_3sigdisc'
    #tag = 'gri_lambda0.3_3sigdisc_copy'
    #imtag = 'gri_lambda0.3_control'
    #tag = 'gri_lambda0.3_control'
    #imtag = 'gri_100k'
    #tag = 'gri_100k_lambda0.3'
    imtag = 'gri'
    tag = 'gri_lambda0.3'

    imarr_fn = f'/scratch/ksf293/anomalies/data/images_h5/images_{imtag}.h5'
    results_dir = f'/scratch/ksf293/anomalies/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading data & residuals")
    imarr = h5py.File(imarr_fn, "r")
    res = h5py.File(results_fn, "a")
    
    NBANDS = 3
    NSIDE = 96
    chunksize = 1000
    print("Creating new dataset")
    if 'reals' in res.keys():
        del res['reals']
    res.create_dataset('reals', (0,NSIDE,NSIDE,NBANDS), maxshape=(None,NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')
    
    reals = imarr['images'][:]
    
    start = 0
    stop = start + chunksize
    n_reals = len(reals)
    print(stop+chunksize, n_reals)
    while True:
        print(f"Computing chunk {start}-{stop}")
        
        reals_chunk = reals[start:stop]
        reals_chunk = reals_chunk.reshape((-1,NSIDE,NSIDE,NBANDS))
        reals_chunk = utils.luptonize(reals_chunk).astype('int')

        addsize = stop-start
        res['reals'].resize(res['reals'].shape[0]+addsize, axis=0)
        res['reals'][-reals_chunk.shape[0]:] = reals_chunk

        if stop==n_reals:
            break
        start = stop
        stop = min(stop+chunksize, n_reals)




    print(res['reals'])
    print(res['reconstructed'])
    imarr.close()
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
