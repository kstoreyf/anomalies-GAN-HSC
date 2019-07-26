import os
import numpy as np

from astropy.io import fits


### Only works for one band right now! Does not combine them yet

tag = 'i60k'
image_dir = f"images_fits/images_{tag}"
out_tag = '_96x96'
size = 96

def main():
    
    npdir = 'imarrs_np'
    if not os.path.isdir(npdir):
        os.mkdir(npdir)

    nparrs = [unpack(fn) for fn in os.listdir(image_dir)]
    indices = [get_idx(fn) for fn in os.listdir(image_dir)]
    np.save(f'{npdir}/hsc_{tag}{out_tag}.npy', nparrs)
    np.save(f'{npdir}/hsc_{tag}{out_tag}_idx.npy', indices)


def unpack(fn):
    #im = fits.getdata(f"{image_dir}/{fn}", 1)
    #print(im.shape)
    with fits.open(f"{image_dir}/{fn}", memmap=False) as hdul:
        im = hdul[1].data
    centh = im.shape[0]/2
    centw = im.shape[1]/2    
    lh, rh = int(centh-size/2), int(centh+size/2)
    lw, rw = int(centw-size/2), int(centw+size/2)
    cropped = im[lh:rh, lw:rw]
    assert cropped.shape[0]==size and cropped.shape[1]==size, f"Wrong size! Still {cropped.shape}"
    return cropped


def get_idx(fn):
    namearr = fn.split('_')
    return int(namearr[2])


if __name__=='__main__':
    main()

