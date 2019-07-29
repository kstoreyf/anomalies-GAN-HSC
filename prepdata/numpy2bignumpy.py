# **************************************************
# * File Name : numpy2bignumpy.py
# * Creation Date : 2019-07-29
# * Created By : kstoreyf
# * Description : Combines batched numpy files into 
#       one large numpy file per band
# **************************************************

import os
import numpy as np

tag = 'i20.0'
image_dir = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarrs_{tag}'
print("Combining numpy arrays...")
arrs = np.concatenate([np.load(f'{image_dir}/{fn}') for fn in os.listdir(image_dir) if 'idx' not in fn])
np.save(f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy', arrs)
print("Image arrays done")

indices = np.concatenate([np.load(f'{image_dir}/{fn}') for fn in os.listdir(image_dir) if 'idx' in fn])
np.save(f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}_idx.npy', indices)
print("Index arrays done")
