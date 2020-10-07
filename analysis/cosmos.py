# **************************************************
# * File Name : cosmos.py
# * Creation Date : 2019-10-16
# * Created By : kstoreyf
# * Description :
# **************************************************

import os
import sys
import numpy as np

import astropy.units as u
from astropy import wcs
from astropy.io import fits
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
from astropy.utils.data import download_file, clear_download_cache
import matplotlib.pyplot as plt
import pandas as pd
import h5py



print("Loading catalog")
cat_cosmos_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv'
if not os.path.isfile(cat_cosmos_fn):
    cat_fn = '/archive/k/ksf293/kavli/anomaly/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
    cat = pd.read_csv(cat_fn)

    print("Cutting out cosmos")
    ra_max = 150.8
    ra_min = 149.4
    dec_max = 2.9
    dec_min = 1.6
    cat_cosmos = cat[(cat['ra_x']<ra_max) & (cat['ra_x']>ra_min)
               & (cat['dec_x']<dec_max) & (cat['dec_x']>dec_min)]
    cat_cosmos.to_csv(cat_cosmos_fn)
else:
    cat_cosmos = pd.read_csv(cat_cosmos_fn)

idxs_cosmos = np.array(cat_cosmos['idx'])
print(idxs_cosmos[:10])
#print(cat_cosmos.index[:10])





## load big h5 files
print("Setting up all data")
tag = 'gri'
data_path = '/scratch/ksf293/kavli/anomaly/data'
res_path = '/scratch/ksf293/kavli/anomaly/results'
imarr_fn = '{}/images_h5/images_{}.h5'.format(data_path, tag)
results_fn = '{}/results_{}.h5'.format(res_path, tag)

## make new h5 files
print("Setting up new cosmos datasets")
imarr_cosmos_fn = '{}/images_h5/images_gri_cosmos_fix2.h5'.format(data_path)
results_cosmos_fn = '{}/results_gri_cosmos_fix2.h5'.format(res_path)
copy_im = True
copy_res = False



#imarr = h5py.File(imarr_fn, 'r')
#im_ix = np.argsort(imarr['idxs'])
#im_ix_fix = im_ix[1:]
#print(im_ix_fix[0])
#print(imarr['idxs'][0])
#print(imarr['idxs'][im_ix_fix[0]])
#print(imarr['object_ids'][im_ix_fix[0]])
#
#imarr.close()
#print(ksdjk)

## copy cosmos data
print("Copying data to cosmos datasets")
if copy_res:
    print("Res")
    res = h5py.File(results_fn, 'r')
    fres_cosmos = h5py.File(results_cosmos_fn,"w")
    idxs_res = res['idxs']
    nres = len(idxs_res)
    ii_res_cosmos = [i for i in range(nres) if idxs_res[i] in idxs_cosmos]
    print(ii_res_cosmos[:100])
    for key in res.keys():
        print(key)
        carr = [res[key][i] for i in ii_res_cosmos]
        if key=='idxs':
            print(carr[:100])
        fres_cosmos.create_dataset(key, data=carr)
    res.close()
    fres_cosmos.close()
    
if copy_im:
    print("Imarr")
    imarr = h5py.File(imarr_fn, 'r')
    fimarr_cosmos = h5py.File(imarr_cosmos_fn,"w")
    idxs_im = imarr['idxs']
    nim = len(idxs_im)-1
    ii_im_cosmos = [i for i in range(nim) if idxs_im[i] in idxs_cosmos]
    for key in imarr.keys():
        print(key)
        carr = [imarr[key][i] for i in ii_im_cosmos]
        fimarr_cosmos.create_dataset(key, data=carr)
    imarr.close()
    fimarr_cosmos.close()

print("Done! Closing up shop")


