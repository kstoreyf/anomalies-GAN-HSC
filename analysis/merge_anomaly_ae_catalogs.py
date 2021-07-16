# **************************************************
# * File Name : merge_anomaly_ae_catalogs.py
# * Creation Date : 2021-07-13
# * Created By : kstoreyf
# * Description :
# **************************************************

import h5py
import numpy as np
import pandas as pd
from astropy.table import Table, join

from caterpillar import catalog


save_fn = '/scratch/ksf293/anomalies/data/hsc_catalogs/anomaly_catalog_hsc_full_ae.fits'
tag = 'gri_lambda0.3'
aenum = 30000
mode = 'reals'
aetag = f'_latent64_{mode}_long_lr1e-4'
savetag = f'_model{aenum}{aetag}'
base_dir = '/scratch/ksf293/anomalies'
results_dir = f'{base_dir}/results'
results_ae_fn = f'{results_dir}/results_ae_{tag}{savetag}.h5'
aeres = h5py.File(results_ae_fn, 'r')

# Load anomaly catalog
print("Loading anomaly catalog")
anom_file = '/scratch/ksf293/anomalies/data/hsc_catalogs/anomaly_catalog_hsc_full.fits'
anom_cat = Table.read(anom_file)

# Add desired info from other query file
print("Loading info catalog")
base_dir = '/scratch/ksf293/anomalies'
info_fn = f'{base_dir}/data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
info_df = pd.read_csv(info_fn, usecols=['idx', 'i_blendedness_abs_flux'])
info_cat = Table.from_pandas(info_df)

print("Joining anomaly and info")
anom_cat = join(anom_cat, info_cat, join_type='left', keys='idx')

print("Adding shape info")
shape_types_to_add = ['i_sdss_shape', 'i_cmodel_exp_ellipse', 'i_cmodel_ellipse', 
                 'r_cmodel_exp_ellipse', 'r_cmodel_ellipse']
for shape_type in shape_types_to_add:
    anom_cat = catalog.moments_to_shape(anom_cat, shape_type=shape_type, update=True)

print("Computing aperature ratios, colors, R_eff")
# compute aperture ratios
anom_cat['r_aperature_ratio'] = np.log10(anom_cat['r_convolvedflux_3_20_flux'] / anom_cat['r_cmodel_flux'])
anom_cat['i_aperature_ratio'] = np.log10(anom_cat['i_convolvedflux_3_20_flux'] / anom_cat['i_cmodel_flux'])

# compute colors
anom_cat['gr_color_cmod'] = (-2.5 * np.log10(anom_cat['g_cmodel_flux']/ anom_cat['r_cmodel_flux']))
anom_cat['ri_color_cmod'] = (-2.5 * np.log10(anom_cat['r_cmodel_flux']/ anom_cat['i_cmodel_flux']))

# log of R_eff
anom_cat['i_cmodel_exp_ellipse_logr'] = np.log10(anom_cat['i_cmodel_exp_ellipse_r'])

# Add AE catalog info
print("Adding AE catalog info")
ae_dataset_names = ['ae_anomaly_scores_sigma','idxs']
column_renames = {'ae_anomaly_scores_sigma': 'ae_score_normalized',
                  'idxs': 'idx'}
ae_column_names = [column_renames[dn] for dn in ae_dataset_names]
ae_data = np.vstack([aeres[cn][:] for cn in ae_dataset_names]).T
ae_cat = Table(ae_data, names=ae_column_names)

print("Joining anomaly and AE")
anom_cat = join(anom_cat, ae_cat, join_type='left', keys='idx')

print("Writing new catalog")
anom_cat.write(save_fn)

print("Done!")
