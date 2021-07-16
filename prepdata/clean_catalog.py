# **************************************************
# * File Name : clean_catalog.py
# * Creation Date : 2019-08-12
# * Created By : kstoreyf
# * Description :
# **************************************************
import time
import numpy as np

import pandas as pd

def main():
    fn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5.csv"
    print("Loading")
    alldata = pd.read_csv(fn)
    print("Cleaning")
    cleandata = clean_data(alldata)
    print("Saving")
    cleandata.to_csv("../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean.csv")


def clean_data(data):
    print("Cleaning data")
    start = time.time()
    bands = ['g','r','i','z','y']
    badflags = ['pixelflags_edge',
                'pixelflags_interpolatedcenter',
                'pixelflags_saturatedcenter',
                'pixelflags_crcenter',
                'cmodel_flag']
    for i in range(len(badflags)):
        for b in range(len(bands)):
            flag = '{}_{}'.format(bands[b], badflags[i])
            data = data[data[flag]==0]
    end = time.time()
    print(f"Cleaned catalog: {len(data)} objects")
    print('Time clean:',end-start)
    return data


if __name__=='__main__':
    main()
