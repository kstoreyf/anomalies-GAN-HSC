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
   
    tag = 'gri_3signorm'
    #tag = 'gri_100k'
    #tag = 'gri_cosmos'
    #tag = 'gri'
    info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'

    results_dir = f'/scratch/ksf293/kavli/anomaly/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading results")
    res = h5py.File(results_fn, "a")
    idxs = [idx.astype(np.uint32) for idx in res['idxs'][:]]

    print("Read in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn, usecols=['object_id', 'idx'], squeeze=True)
    info_df = info_df.set_index('idx')
    object_ids = [info_df['object_id'].loc[idx].astype(np.uint64) for idx in idxs]
    
    print("Creating new dataset")
    del res["object_ids"] #make sure to run this line if it exists bc you messed up!
    res.create_dataset("object_ids", data=object_ids, dtype='uint64')
    
    print("While we're at it, fix idxs -> integer")
    del res['idxs']
    res.create_dataset("idxs", data=idxs, dtype='uint32')

    res.close() 
    print("Done")


if __name__=='__main__':
    main()
