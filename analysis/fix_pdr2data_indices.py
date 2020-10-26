# **************************************************
# * File Name : fix_pdr2data_indices.py
# * Creation Date : 2020-10-06
# * Created By : kstoreyf
# * Description :
# **************************************************
import pandas as pd


def main():

    info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
    #info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv'

    print("Read in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn)
    
    info_df.rename({'Unnamed: 0': 'idx'}, axis=1, inplace=True)
    info_df.drop('Unnamed: 0.1', axis=1, inplace=True)
    if 'cosmos' in info_fn:
        info_df.rename({'Unnamed: 0.1.1.1': 'idx_orig'}, axis=1, inplace=True)
        info_df.drop('Unnamed: 0.1.1', axis=1, inplace=True)
    else:
        info_df.rename({'Unnamed: 0.1.1': 'idx_orig'}, axis=1, inplace=True)

    info_df = info_df.set_index('idx')
    # now when load it in, use: info_df = info_df.set_index('idx')

    info_df.to_csv(info_fn)

if __name__=='__main__':
    main()
