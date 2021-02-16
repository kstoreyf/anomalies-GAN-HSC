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

    #tag = 'gri_lambda0.3_3sigdisc'
    tag = 'gri_lambda0.3_1.5sigdisc'
    #tag = 'gri_lambda0.3_control'
    #tag = 'gri_1k_lambda0.3'

    base_dir = '/scratch/ksf293/anomalies'
    tar_filename = f'../thumbnails/thumbnails_galaxyzoo_{tag}.tar.gz'
    thumb_dir = f'{base_dir}/thumbnails/thumbnails_galaxyzoo_{tag}'
    print(f"Thumbnail directory: {thumb_dir}")
    if not os.path.isdir(thumb_dir):
        os.makedirs(thumb_dir)

    print(f"Loading and results with tag {tag}")
    results_fn = f'{base_dir}/results/results_{tag}.h5'
    res = h5py.File(results_fn, 'r')

    #write_thumbnails(res, thumb_dir)
    #tar_thumbnails(tar_filename, thumb_dir)
    write_info_file(res, tag)

    res.close()
    print("Done!")


def tar_thumbnails(tar_filename, thumb_dir):
    print("Writing tarball")
    with tarfile.open(tar_filename, "w:gz") as tar:
        tar.add(thumb_dir, arcname=os.path.basename(thumb_dir))
    print("Tar'ed!")

def write_thumbnails(res, thumb_dir):
    print("Writing thumbnails")
    reals = res['reals']
    n_ims = len(reals)

    print(f'Saving {n_ims} images as jpgs')
    for i in range(n_ims):
        im = Image.fromarray(reals[i])
        idx = res['idxs'][i]
        objid = res['object_ids'][i]
        score = res['disc_scores_sigma'][i]
        save_fn = f"{thumb_dir}/thumbnail_idx{idx}_objectid{objid}_discscore{score:.3f}.jpg"
        im.save(save_fn)
    print("Written!")


def write_info_file(res, tag):
    print("Writing associated info file") 
    # column names that will need changing 
    column_names = {'ra_x':'ra', 
                    'dec_x': 'dec',
                    'disc_scores': 'disc_scores_raw'}
    
    idxs = res['idxs'][:]
    score_names = ['disc_scores_sigma', 'disc_scores']
    score_data = np.empty((len(idxs), len(score_names)))
    for i, sn in enumerate(score_names):
        score_data[:,i] = res[sn][:]
    score_df = pd.DataFrame(data=score_data, index=idxs, columns=score_names)
    score_df.index.name = 'idx'

    info_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
    print("Reading in info file {}".format(info_fn))
    info_df = pd.read_csv(info_fn, usecols=['idx', 'object_id', 'ra_x', 'dec_x'], squeeze=True)
    info_df = info_df.set_index('idx')

    # Combine dfs
    combined_df = pd.merge(score_df, info_df, on='idx', how='left')
    combined_df.rename(columns=column_names,inplace=True)
    combined_df = combined_df[['object_id', 'ra', 'dec', 'disc_scores_sigma', 'disc_scores_raw']]
    combined_df.index = combined_df.index.astype(np.uint32) #i'm not sure why this became a float but fixing
    combined_df.sort_index(inplace=True)
    print(combined_df)

    save_fn = f'../thumbnails/info_galaxyzoo_{tag}.csv'
    combined_df.to_csv(save_fn)
    print("Info file written!")

if __name__=='__main__':
        main()
