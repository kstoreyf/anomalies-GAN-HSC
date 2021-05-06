# ******************************************************
# * File Name : check_fix_object_ids_formatting
# * Creation Date : 
# * Created By : kstoreyf
# * Description : Check and fix the object IDs and their 
#                 formatting associated with images
# ******************************************************

import numpy as np
import pandas as pd
import h5py
import glob


def main():

    base_dir = '/scratch/ksf293/anomalies'
   
    info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'

    hf_fns = []
    #restags = ['gri_lambda0.3_control']#, 'gri_lambda0.3']
    #restags = ['gri_100k_lambda0.3']
    imtags = ['gri_3sig']
    #hf_fns += [f'{base_dir}/results/results_{tag}.h5' for tag in restags]
    #hf_fns += glob.glob(f'{base_dir}/results/results_*.h5')
    #hf_fns += [f'{base_dir}/data/images_h5/images_{tag}.h5' for tag in imtags]
    hf_fns += glob.glob(f'{base_dir}/data/images_h5/images_*.h5')
    print(hf_fns)

    print("Read in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn, usecols=['object_id', 'idx'], squeeze=True)
    info_df = info_df.set_index('idx')

    for hf_fn in hf_fns:
        check_objids(hf_fn, info_df, fix=False)


def check_objids(hf_fn, info_df, base_dir='/scratch/ksf293/anomalies', fix=False):

    print(f"Reading file {hf_fn}")
    if fix:
        mode = 'a'
    else:
        mode = 'r'
    hf = h5py.File(hf_fn, mode)
    
    if 'idxs' not in hf.keys():
        print("NO IDXS! returning")
        hf.close()
        return
    idxs_hf = hf['idxs'][:]
    
    if idxs_hf.dtype!=np.uint32:
        print(f"idxs_hf bad type! was {idxs_hf.dtype}")
        if fix:
            idxs_hf = fix_idx_type(hf, idxs_hf)
        else:
            idxs_hf = [idx.astype(np.uint32) for idx in idxs_hf]

    object_ids_hf = np.array(hf['object_ids'][:])
    if object_ids_hf.dtype!=np.uint64:
        print(f"objids bad type! was {object_ids_hf.dtype}")
        if fix:
            object_ids_hf = fix_object_id_type(hf, object_ids_hf)
        else:
            object_ids_hf = [object_ids_hf.astype(np.uint64) for objid in object_ids_hf]

    object_ids_info = np.array([info_df['object_id'].loc[idx].astype(np.uint64) for idx in idxs_hf])

    try:
        np.testing.assert_equal(object_ids_hf, object_ids_info)
        print(f"Passed! ({hf_fn})")
    except:
        print(f"Failed! ({hf_fn})")
        if fix:
            fix_object_ids(hf, object_ids_info)
    
    hf.close() 


def fix_object_ids(hf, object_ids_correct):
    print("Fixing object ids")
    if "object_ids" in hf.keys():
        del hf["object_ids"]
    hf.create_dataset("object_ids", data=object_ids_correct, dtype='uint64')

def fix_idx_type(hf, idxs_hf):
    print("Fixing idx type")
    idxs_correct = [idx.astype(np.uint32) for idx in idxs_hf]
    del hf['idxs']
    hf.create_dataset("idxs", data=idxs_correct, dtype='uint32')
    return idxs_correct

def fix_object_id_type(hf, object_ids_hf):
    print("Fixing object_id type")
    object_ids_correct = [object_ids_hf.astype(np.uint64) for objid in object_ids_hf]
    if "object_ids" in hf.keys():
        del hf['object_ids']
    hf.create_dataset("object_ids", data=object_ids_correct, dtype='uint64')
    return object_ids_correct

if __name__=='__main__':
    main()
