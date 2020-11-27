# ******************************************************
# * File Name : compute_iresiduals_rgb.py
# * Creation Date : 2020-07-17
# * Created By : kstoreyf
# * Description : Computes the image residuals from the
#                 real & reconstructeds, for RGB images
# ******************************************************

import numpy as np
import pandas as pd
import h5py


def main():
   
    #tag = 'gri_3signorm'
    #tag = 'gri_3sig'
    #tag = 'gri_100k'
    #tag = 'gri_cosmos'
    tag = 'gri'
    info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'

    base_dir = '/scratch/ksf293/kavli/anomaly'
    imarr_fn = f'{base_dir}/data/images_h5/images_{tag}.h5'

    print("Loading imarr")
    imarr = h5py.File(imarr_fn, "a")
    idxs = [idx.astype(np.uint32) for idx in imarr['idxs'][:]]

    print("First fix idxs -> integer")
    del imarr['idxs']
    imarr.create_dataset("idxs", data=idxs, dtype='uint32')

    print("Read in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn, usecols=['object_id', 'idx'], squeeze=True)
    info_df = info_df.set_index('idx')
    object_ids = [info_df['object_id'].loc[idx].astype(np.uint64) for idx in idxs]
    
    print("Creating new object_id dataset")
    del imarr["object_ids"]
    imarr.create_dataset("object_ids", data=object_ids, dtype='uint64')
    
    imarr.close() 
    print("Done")


if __name__=='__main__':
    main()
