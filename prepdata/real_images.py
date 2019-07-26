import numpy as np

import tflib_py3.save_images as saver


tag = 'i1k_96x96_norm'
nside = 96
ng = 128
imarr96 = np.load(f'imarrs_np/hsc_{tag}.npy')
ims = imarr96[:ng]
ims.reshape((ng, 96, 96))
saver.save_images(ims, f"thumbnails/real_{tag}.png")
