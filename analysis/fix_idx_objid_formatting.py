# **************************************************
# * File Name : fix_idx_objid_formatting.py
# * Creation Date : 2021-01-20
# * Created By : kstoreyf
# * Description :
# **************************************************
import h5py
import numpy as np

def main():
    
    tag = 'gri_lambda0.3_1.5sigdisc'
    #tag = 'gri_3sigdisc'

    print(f"Loading and results with tag {tag}")
    base_dir = '/scratch/ksf293/kavli/anomaly'
    results_fn = f'{base_dir}/results/results_{tag}.h5'
    res = h5py.File(results_fn, 'r+') 
    
    # Fix idxs
    idxs = res['idxs'][:]
    idxs_fixed = idxs.astype('int')
    del res['idxs']
    res.create_dataset("idxs", data=idxs_fixed, dtype='uint32')

    # Fix object ids
    obj_ids = res['object_ids'][:]
    obj_ids_fixed = obj_ids.astype('int')
    del res['object_ids']
    res.create_dataset("object_ids", data=obj_ids_fixed, dtype='uint64')

    res.close()
    print("Done!")

if __name__=='__main__':
    main()
