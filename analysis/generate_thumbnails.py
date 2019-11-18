import os
import numpy as np

import h5py
from imageio import imwrite

import utils


thumb_dir = '../thumbnails/hsc/hsc_gri_3sig_top100-200'
print(f"Thumbnail directory: {thumb_dir}")
if not os.path.isdir(thumb_dir):
    os.makedirs(thumb_dir)

tag = 'gri_3sig'
print(f"Loading images and results with tag {tag}")
imarr_fn = '../data/images_h5/images_{}.h5'.format(tag)
results_fn = '../results/results_{}.h5'.format(tag)
imarr = h5py.File(imarr_fn, 'r')
res = h5py.File(results_fn, 'r')

scores = res['anomaly_scores']
idxs = res['idxs']
object_ids = res['object_ids']

print("Getting subset of objects")
# get top 100 scoring objects
isort = np.argsort(scores)
idxs_sorted = np.array(idxs)[isort]
idxs_tothumb = idxs_sorted[-200:-100]

print("Make lookup dicts for indices")
idx2imloc = {}
for i in range(len(imarr['idxs'])):
    idx2imloc[imarr['idxs'][i]] = i
idx2resloc = {}
for i in range(len(res['idxs'])):
    idx2resloc[res['idxs'][i]] = i

print("Saving thumbnails")
for idx in idxs_tothumb:
    idx = int(idx)
    #obj = cat.loc[idx]
    #objid = obj['object_id']

    imloc = idx2imloc[idx]
    resloc = idx2resloc[idx]
    im = imarr['images'][imloc]
    objid = int(object_ids[resloc])
    scoreint = int(scores[resloc])

    save_fn = f"{thumb_dir}/hsc_idx{idx}_objectid{objid}_score{scoreint}.png"
    im = utils.luptonize(im)
    imwrite(save_fn, im)

print("Closing up shop")
res.close()
imarr.close()
