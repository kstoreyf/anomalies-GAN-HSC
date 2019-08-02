import numpy as np
from imageio import imwrite

import tflib.save_images as saver


tag = 'i20.0'
nside = 96
ng = 1
#idx = 77817

idx = 7819
#imarr96 = np.load(f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy')
tag = 'i20.0_norm_100k'
savetag = f'_{idx}'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
save_fn = f"../thumbnails/real_{tag}{savetag}.png"
imarr = np.load(imarr_fn)

ims = imarr[idx]
print(ims.shape)

imwrite(save_fn, ims)
#ims.reshape((96, 96))
#saver.save_images(ims, f"../thumbnails/real_{tag}{savetag}.png")
