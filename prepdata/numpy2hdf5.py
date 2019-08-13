# **************************************************
# * File Name : numpy2hdf5.py
# * Creation Date : 2019-08-11
# * Created By : kstoreyf
# * Description :
# **************************************************
import os
import h5py
import numpy as np
import pandas as pd

#path = '/tmp/out.h5'
save_dir = '/scratch/ksf293/kavli/anomaly/data/images_h5'
savetag = 'gri'
save_fn = f'{save_dir}/images_{savetag}.h5'
load_dir = '/scratch/ksf293/kavli/anomaly/data/images_np'
load_tags = ['g20.0', 'r20.0', 'i20.0']
#nfiles = 943
nbands = len(load_tags)
nfiles = 943
#ntotal = 942781

#nfiles = 10
#ntotal = 10*1000
#indices = np.random.choice(ntotal, size=1000, replace=False)

print("Loading catalog")
catfn = "../data/hsc_catalogs/pdr2_wide_icmod_20.0-20.5_clean.csv"
cat = pd.read_csv(catfn)
ntotal = len(cat)
print(ntotal)
indices = np.random.choice(ntotal, size=ntotal, replace=False)
print("Writing h5 file")


with h5py.File(save_fn, "w") as f:
    f.create_dataset('images', (0,96,96,nbands), maxshape=(None,96,96,nbands),
                            chunks=(1,96,96,nbands)) #chunks=(10**3,96,96,nbands))
    f.create_dataset('idxs', (1,), maxshape=(None,),
                                chunks=True)
    f.create_dataset('object_ids', (1,), maxshape=(None,),
                                    chunks=True)
    count = 0
    for filenum in range(nfiles):
        print(filenum)
        im_arrs = []
        idx_arrs = []
        for j in range(len(load_tags)):
            ltag = load_tags[j]
            ims = np.load(f'{load_dir}/imarrs_{ltag}/hsc_{ltag}_{filenum}.npy')
            idxs = np.load(f'{load_dir}/imarrs_{ltag}/hsc_{ltag}_{filenum}_idx.npy')
            
            imsidxs = zip(*[[imi, idxi] for imi, idxi in zip(ims, idxs) if idxi in indices])
            try:
                im_sample, idx_sample = imsidxs
            except ValueError:
                break
            im_arrs.append(list(im_sample))
            idx_arrs.append(list(idx_sample))

        if not im_arrs:
            continue

        #if f['images'].shape[0]==1:
        #    addsize = len(im_arrs[0])-1
        #else:
        #    addsize = len(im_arrs[0])
        addsize = len(im_arrs[0])
        f['images'].resize(f['images'].shape[0]+addsize, axis=0)
        f['idxs'].resize(f['idxs'].shape[0]+addsize, axis=0)
        f['object_ids'].resize(f['object_ids'].shape[0]+addsize, axis=0)
        for i in range(len(idx_arrs[0])):
            idx = idx_arrs[0][i]
            #print(idx)
            #if idx not in indices:
            #    continue

            im = np.empty((96,96,nbands))
            for jj in range(len(load_tags)):
                #print(idx_arrs[jj])
                #print(np.where(idx_arrs[jj]==idx)[0])
                #im[:,:,jj] = im_arrs[jj][np.where(idx_arrs[jj]==idx)[0]]
                im[:,:,jj] = im_arrs[jj][idx_arrs[jj].index(idx)]

            #nrows = arr.shape[0]
            #print(arr.shape)
            #dset.resize(dset.shape[0]+nrows, axis=0)   
            #dset[-nrows:,:,:] = arr
            f['images'][count, ...] = im
            f['idxs'][count, ...] = idx
            obj_id = cat['object_id'][idx]
            f['object_ids'][count, ...] = int(obj_id)
            count += 1
        print(f['images'].shape)
