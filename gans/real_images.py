# ******************************************************
# * File Name : real_images.py
# * Creation Date : 2019-08-02
# * Created By : kstoreyf
# * Description : Generates image thumbnails or batches.
# ******************************************************
import numpy as np
from imageio import imwrite

import tflib.save_images as saver
import tflib.datautils as datautils

def main():
    anoms = [102009] #i20.0_norm
    #idx = 15850
    #tag = 'i20.0_norm'
    tag = 'gri_1k'
    #imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
    
    savetag = '_nolupnorm'
    save_fn = f"../thumbnails/real_{tag}{savetag}.png"
    #imarr = np.load(imarr_fn)
    imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'
    imarr = datautils.load(imarr_fn)
    #idx = [0,1]
    #make_thumbnail(imarr, idx, tag)
    make_batch(imarr, save_fn)

def make_thumbnail(imarr, idxs, tag):
    if type(idxs)==int:
        idxs = [idxs]
    for idx in idxs:
        ims = imarr[idx]
        save_fn = f"../thumbnails/real_{tag}_{idx}.png"
        imwrite(save_fn, np.flip(ims, 2))


def make_batch(imarr, save_fn, n=128):
    ims = imarr[:n]
    print(ims.shape)
    ims.reshape((n, 96, 96, -1))
    ims = np.array([normalize(im) for im in ims])
    print(ims.shape)
    #ims = np.flip(ims, 3)
    saver.save_images(ims, save_fn, luptonize=False, unnormalize=False)

def norm0to1(a):
    return (a - np.min(a))/np.ptp(a)

def normalize(d):
    #if d.ndim>2:
    #    for i 
    d = np.arcsinh(d)
    d = norm0to1(d)
    return(d)

if __name__=='__main__':
    main()
