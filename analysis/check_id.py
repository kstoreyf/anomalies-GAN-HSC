import numpy as np
import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
from astropy.visualization import make_lupton_rgb
import astropy.units as u
import pandas as pd
import h5py
from imageio import imwrite

import unagi
from unagi import config
from unagi import hsc
from unagi.task import hsc_tricolor, hsc_cutout

from utils import luptonize


tag = 'gri_cosmos'
#tag = 'gri_3sig'
#idx_tocheck = 937431
#idx_tocheck = 406992
idx_tocheck = 935811
save_dir = '../thumbnails/id_check'

#cat_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_cosmos.csv'
cat_fn = '../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean_more.csv'
cat = pd.read_csv(cat_fn)
cat = cat.set_index('Unnamed: 0')

### H5PY FILES ###
print("Checking image in h5py files - imarr, results")

imarr_fn = '../data/images_h5/images_{}.h5'.format(tag)
results_fn = '../results/results_{}.h5'.format(tag)

imarr = h5py.File(imarr_fn, 'r')
res = h5py.File(results_fn, 'r')

scores = res['anomaly_scores']
idxs = res['idxs']
object_ids = res['object_ids']

idx2imloc = {}
for i in range(len(imarr['idxs'])):
    idx2imloc[imarr['idxs'][i]] = i
    
idx2resloc = {}
for i in range(len(res['idxs'])):
    idx2resloc[res['idxs'][i]] = i

### IMARR ###
plt.figure(figsize=(4,4))
ax = plt.gca()
loc = idx2imloc[idx_tocheck]
im = luptonize(imarr['images'][loc])
ax.imshow(im)
ax.set_title(idx_tocheck)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"{save_dir}/thumbnail_{tag}_{idx_tocheck}_imarr.png")

### RECONSTRUCTED ###
plt.figure(figsize=(4,4))
ax = plt.gca()
loc = idx2resloc[idx_tocheck]
recon = res['reconstructed'][loc]
ax.imshow(recon)
ax.set_title(idx_tocheck)
ax.set_xticks([])
ax.set_yticks([])
plt.savefig(f"{save_dir}/thumbnail_{tag}_{idx_tocheck}_reconstructed.png")

### CHECK HSC CATALOG ###
print("Checking in HSC catalog")

objx = cat.loc[idx_tocheck]
rax = objx['ra_x']
decx = objx['dec_x']
print("RA:", rax, "dec:", decx)

print("Getting HSC archive")
pdr2_wide = hsc.Hsc(dr='pdr2', rerun='pdr2_wide', config_file='../cred.dat')

filters = ['g','r','i']
ang_size_h = 10*u.arcsec
ang_size_w = 10*u.arcsec

coord = SkyCoord(rax, decx, frame='icrs', unit='deg')
s_ang = [ang_size_h, ang_size_w]

print("Querying HSC")
cutout_rgb, cutout_wcs = hsc_tricolor(
    coord, cutout_size=s_ang, filters=filters, verbose=False, 
    save_rgb=False, save_img=False, use_saved=False, archive=pdr2_wide)

print("Plotting")
plt.figure(figsize=(4,4))
ax = plt.gca()
ax.imshow(cutout_rgb, origin='upper')
ax.set_xticks([])
ax.set_yticks([])
ax.set_title(idx_tocheck)
plt.savefig(f"{save_dir}/thumbnail_{tag}_{idx_tocheck}_hsc.png")
print("Done!")
