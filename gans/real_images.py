import numpy as np
from imageio import imwrite

import tflib.save_images as saver


def main():
    tag = 'i20.0'
    idx = 15850
    #imarr96 = np.load(f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy')
    tag = 'i20.0_norm_100k'
    savetag = f'_{idx}'
    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
    save_fn = f"../thumbnails/real_{tag}{savetag}.png"
    imarr = np.load(imarr_fn)
    make_thumbnail(imarr, idx, save_fn)

def make_thumbnail(idx, imarr, savefn):
    ims = imarr[idx]
    imwrite(save_fn, ims)


def make_batch(save_fn, n=128):
    ims = imarr[:n]
    ims.reshape((-1, 96, 96))
    saver.save_images(ims, save_fn)

if __name__=='__main__':
    main()
