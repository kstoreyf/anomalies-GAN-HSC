import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import pandas as pd
import h5py


#tag = 'gri_cosmos'
tag = 'gri_3sig'
#idx_tocheck = 937431
#idx_tocheck = 406992
#idx_tocheck = 935055
#idx_tocheck = 936569
idx_tocheck = 941128

#cat_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv'
cat_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
cat = pd.read_csv(cat_fn).set_index('idx')


### CHECK HSC CATALOG ###
print("Checking in HSC catalog")

objx = cat.loc[idx_tocheck]
rax = objx['ra_x']
decx = objx['dec_x']
print("RA:", rax, "dec:", decx)
for key in objx.keys():
    print("{}: {}".format(key, objx[key]))
