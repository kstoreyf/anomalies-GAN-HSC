import os
import numpy as np

from astropy.io import fits


### Only works for one band right now! Does not combine them yet

fits_tag = 'i20.0'
image_dir = f"/scratch/ksf293/kavli/anomaly/data/images_fits/images_{fits_tag}"
out_tag = 'i20.0_800k'
size = 96

def main():
    
    npdir = '/scratch/ksf293/kavli/anomaly/data/imarrs_np'
    if not os.path.isdir(npdir):
        os.mkdir(npdir)

    #add_dir = f"/scratch/ksf293/kavli/anomaly/imarrs_np"
    #add_tag = "i60k_96x96"
    add_tag = None

    nparrs = [unpack(fn) for fn in os.listdir(image_dir)]
    indices = [get_idx(fn) for fn in os.listdir(image_dir)]

    if add_tag:
        nparrs_add = np.load(f"{add_dir}/hsc_{add_tag}.npy")
        indices_add = np.load(f"{add_dir}/hsc_{add_tag}_idx.npy")
        print("Concatenating lists")
        nparrs = np.concatenate((nparrs, nparrs_add))
        indices = np.concatenate((indices, indices_add))

    np.save(f'{npdir}/hsc_{out_tag}.npy', nparrs)
    np.save(f'{npdir}/hsc_{out_tag}_idx.npy', indices)


def unpack(fn):
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

