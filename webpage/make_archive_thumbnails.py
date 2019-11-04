import numpy as np
import os
import glob
import re

from imageio import imwrite
from astropy.visualization import make_lupton_rgb
from astropy.io import fits


tag = 'cosmos_redextended'
thumb_dir = f'../thumbnails/cosmos_targets/cosmos_1sig_interesting/{tag}_archive'
fits_dir = f'fits_images/{tag}'
info_table = f'tables/{tag}_archive.dat'

filt_dict = {'COSMOS': ['gp', 'rp', 'ip'], 'acs': ['I']}
pattern_dict = {'COSMOS': 'COSMOS.', 'acs': 'acs_'}

fits_files = glob.glob(fits_dir+'/*')
info = np.loadtxt(info_table, skiprows=2)
idxs = info[:,0]
print(idxs)

nobjs = len(info)
if not os.path.isdir(thumb_dir):
	os.makedirs(thumb_dir)

def luptonize(x, rgb_q=15, rgb_stretch=0.5, rgb_min=0):
	if x.ndim==3:
		x = make_lupton_rgb(x[:,:,2], x[:,:,1], x[:,:,0],
					  Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
	elif x.ndim==4:
		x = np.array([make_lupton_rgb(xi[:,:,2], xi[:,:,1], xi[:,:,0],
					  Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
					  for xi in x])
	else:
		raise ValueError(f"Wrong number of dimensions! Gave {x.ndim}, need 3 or 4")
	return x

#1-indexed
for idx_archive in range(1,nobjs+1):
	idx = int(idxs[idx_archive-1])
	print(idx_archive, idx)
	for dataset in filt_dict.keys():
		im = []
		for filt in filt_dict[dataset]:
			idxa = str(idx_archive).zfill(4)
			pattern = '{}_.*_{}{}.*.fits'.format(idxa, pattern_dict[dataset], filt)
			for f in fits_files:
				match = re.search(pattern, f)
				if match:
					break
			if not match:
				continue
				#raise ValueError('No file with that index and filter!')
			fits_fn = match.group(0)
			hdul = fits.open(f'{fits_dir}/{fits_fn}')
			im.append(np.transpose(hdul[0].data))
		
		if len(im) != len(filt_dict[dataset]):
			continue

		im = np.array(im).T
		if im.shape[-1]==3:
			im = luptonize(im, rgb_stretch=10)
		if im.shape[-1]==1:
			im = np.arcsinh(im)
			immean = np.mean(np.array(im))
			imstd = np.std(np.array(im))
			high = immean + 5*imstd
			im = np.clip(im, None, high)
			im = (im - np.min(im))/np.ptp(im)
		save_fn = f"{thumb_dir}/cosmos_{dataset}_idx{idx}.png"
		imwrite(save_fn, im)



