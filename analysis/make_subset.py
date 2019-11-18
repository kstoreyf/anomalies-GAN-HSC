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


data_path = '/scratch/ksf293/kavli/anomaly/data'
res_path = '/scratch/ksf293/kavli/anomaly/results'

print("Loading catalog")
cat_fn = '{}/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'.format(data_path)
cat = pd.read_csv(cat_fn)

## load big h5 files
print("Setting up all data")
tag = 'gri'
imarr_fn = '{}/images_h5/images_{}.h5'.format(data_path, tag)
results_fn = '{}/results_{}.h5'.format(res_path, tag)

## make new h5 files
print("Setting up new datasets")
imarr_cosmos_fn = '{}/images_h5/images_gri_3sig.h5'.format(data_path)
results_cosmos_fn = '{}/results_gri_3sig.h5'.format(res_path)
copy_im = True
copy_res = False

print("Getting indices")
res = h5py.File(results_fn, 'r')
idxs_res = res['idxs']
scores_res = res['anomaly_scores']
mean = np.mean(scores_res)
std = np.std(scores_res)
thresh = mean+3*std
idxs_tocopy = [idxs_res[i] for i in range(len(idxs_res)) if scores_res[i]>thresh]
print("Identified {} objects to copy".format(len(idxs_tocopy)))
res.close()

## copy cosmos data
print("Copying data to cosmos datasets")
if copy_res:
    print("Res")
    res = h5py.File(results_fn, 'r')
    fres_cosmos = h5py.File(results_cosmos_fn,"w")
    idxs_res = res['idxs']
    nres = len(idxs_res)
    ii_res_cosmos = [i for i in range(nres) if idxs_res[i] in idxs_tocopy]
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
    ii_im_cosmos = [i for i in range(nim) if idxs_im[i] in idxs_tocopy]
    for key in imarr.keys():
        print(key)
        carr = [imarr[key][i] for i in ii_im_cosmos]
        fimarr_cosmos.create_dataset(key, data=carr)
    imarr.close()
    fimarr_cosmos.close()

print("Done! Closing up shop")


