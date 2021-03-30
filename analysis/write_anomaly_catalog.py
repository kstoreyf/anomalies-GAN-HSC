# **************************************************
# * File Name : write_galaxyzoo_thumbnails.py
# * Creation Date : 2021-01-20
# * Created By : kstoreyf
# * Description :
# **************************************************
import os
import h5py
import numpy as np
from PIL import Image
import pandas as pd
import tarfile


def main():

    tag = 'gri_lambda0.3_3sigdisc'
    savetag = '_3sigmadiscfilter'
    #tag = 'gri_lambda0.3_1.5sigdisc'
    #tag = 'gri_lambda0.3_control'
    #tag = 'gri_lambda0.3'
    #savetag = ''

    base_dir = '/scratch/ksf293/anomalies'
    save_fn = f'{base_dir}/results/anomaly_catalogs/anomaly_catalog_hsc{savetag}.csv'

    print(f"Loading and results with tag {tag}")
    results_fn = f'{base_dir}/results/results_{tag}.h5'
    
    info_fn = f'{base_dir}/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
    write_anomaly_catalog(results_fn, info_fn, save_fn)
    print(f"Saved to {save_fn}")

    print("Done!")


def write_anomaly_catalog(results_fn, info_fn, save_fn):
    print("Writing anomaly catalog") 
    # column names 
    column_names = {'object_id': 'HSC_object_id',
                    'ra_x':'ra', 
                    'dec_x': 'dec',
                    'disc_scores_sigma': 'discriminator_score_normalized',
                    #'disc_scores': 'discriminator_score_raw',
                    'gen_scores_sigma': 'generator_score_normalized',
                    #'gen_scores': 'generator_score_raw',
                    'anomaly_scores_sigma': 'combined_score_normalized',
                    #'anomaly_scores': 'combined_score_raw'
                    }
    
    res = h5py.File(results_fn, 'r')
    idxs = res['idxs'][:]
    # names in res object
    #score_names = ['disc_scores_sigma', 'disc_scores', 'gen_scores_sigma', 'gen_scores', 'anomaly_scores_sigma', 'anomaly_scores']
    score_names = [cn for cn in column_names.keys() if 'scores' in cn]
    score_data = np.empty((len(idxs), len(score_names)))
    for i, sn in enumerate(score_names):
        score_data[:,i] = res[sn][:]
    score_df = pd.DataFrame(data=score_data, index=idxs, columns=score_names)
    score_df.index.name = 'idx'

    print("Reading in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn, usecols=['idx', 'object_id', 'ra_x', 'dec_x'], squeeze=True)
    info_df = info_df.set_index('idx')

    # Combine dfs
    combined_df = pd.merge(score_df, info_df, on='idx', how='left')
    combined_df.rename(columns=column_names,inplace=True)
    #combined_df = combined_df[['object_id', 'ra', 'dec', 'disc_scores_sigma', 'disc_scores_raw']]
    combined_df = combined_df[column_names.values()]
    combined_df.index = combined_df.index.astype(np.uint32) #i'm not sure why this became a float but fixing
    combined_df.sort_index(inplace=True)
    print(combined_df)

    combined_df.to_csv(save_fn)
    print("Anomaly catalog  written!")

if __name__=='__main__':
    main()
