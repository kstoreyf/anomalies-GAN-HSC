# **************************************************
# * File Name : build_idxdicts.py
# * Creation Date : 2020-09-20
# * Created By : kstoreyf
# * Description :
# **************************************************

import h5py
import numpy as np


build_imdict = False
build_resdict = True
#tag = 'gri_cosmos'
#tag = 'gri_3sig'
imtag = 'gri'
tag = 'gri_lambda0.3'
base_dir = '/scratch/ksf293/kavli/anomaly'
#base_dir = '/archive/k/ksf293/kavli/anomaly'

if build_imdict:
    print("Building image dict")
    imarr_fn = '{}/data/images_h5/images_{}.h5'.format(base_dir,  imtag)
    imdict_fn = '../data/idxdicts_h5/idx2imloc_{}.npy'.format(imtag)
    imarr = h5py.File(imarr_fn, 'r')
    idx2imloc = {}
    for i in range(len(imarr['idxs'])):
        if i%100000==0: 
            print(i)
        idx2imloc[imarr['idxs'][i]] = i
    np.save(imdict_fn, idx2imloc)

if build_resdict:
    print("Building results dict")
    results_fn = '{}/results/results_{}.h5'.format(base_dir, tag)
    resdict_fn = '../data/idxdicts_h5/idx2resloc_{}.npy'.format(tag)
    res = h5py.File(results_fn, 'r')

    idx2resloc = {}
    for i in range(len(res['idxs'])):
        if i%100000==0: 
            print(i)
        idx2resloc[res['idxs'][i]] = i
    np.save(resdict_fn, idx2resloc)
