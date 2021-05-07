# **************************************************
# * File Name : merge_catalogs.py
# * Creation Date :
# * Created By : kstoreyf
# * Description :
# **************************************************
import time
import numpy as np

import pandas as pd
import unagi
from unagi.catalog import moments_to_shape

def main():
    get_radii()

def get_radii():
    cat_fn = "/scratch/ksf293/kavli/anomaly/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv"
    cat = pd.read_csv(cat_fn)
    print("Getting shapes")
    print(cat.columns)
    shape_type = 'i_cmodel_ellipse'
    rad, ell, theta = moments_to_shape(cat, shape_type=shape_type, update=False)
    cat[f'{shape_type}_radius'] = rad
    cat[f'{shape_type}_ellipticity'] = ell
    cat[f'{shape_type}_theta'] = theta
    print(cat.columns)
    print("saving")
    cat.to_csv(cat_fn)


def merge():
    clean_fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean.csv"
    more_fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_more.csv"
    print("Loading")
    clean = pd.read_csv(clean_fn)
    more = pd.read_csv(more_fn)

    print('merging')
    merged = pd.merge(clean, more, how='left', on='object_id')

    print("saving")
    merged.to_csv("/scratch/ksf293/kavli/anomaly/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv")


if __name__=='__main__':
    main()
