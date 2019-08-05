# ******************************************************
# * File Name : real_images.py
# * Creation Date : 2019-08-02
# * Created By : kstoreyf
# * Description : Generates image thumbnails or batches.
# ******************************************************
import numpy as np
from imageio import imwrite

import tflib.save_images as saver


def main():
    anoms = [102009] #i20.0_norm
    #idx = 15850
    tag = 'i20.0_norm'
    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
    #savetag = ''
    #save_fn = f"../thumbnails/real_{tag}{savetag}.png"
    imarr = np.load(imarr_fn)
    make_thumbnail(imarr, anoms, tag)

def make_thumbnail(imarr, idxs, tag):
    if type(idxs)==int:
        idxs = [idxs]
    for idx in idxs:
        ims = imarr[idx]
        save_fn = f"../thumbnails/real_{tag}_{idx}.png"
        imwrite(save_fn, ims)


def make_batch(save_fn, n=128):
    ims = imarr[:n]
    ims.reshape((-1, 96, 96))
    saver.save_images(ims, save_fn)

if __name__=='__main__':
    main()
