import numpy as np

import tflib.save_images as saver


tag = 'i20.0_19k_norm'
nside = 96
ng = 128
imarr96 = np.load(f'/scratch/ksf293/kavli/anomaly/data/imarrs_np/hsc_{tag}.npy')
ims = imarr96[:ng]
ims.reshape((ng, 96, 96))
saver.save_images(ims, f"../thumbnails/real_{tag}.png")
