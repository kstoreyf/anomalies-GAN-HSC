# ******************************************************
# * File Name : compute_residuals_rgb.py
# * Creation Date : 2020-07-17
# * Created By : kstoreyf
# * Description : Computes the image residuals from the
#                 real & reconstructeds, for RGB images
# ******************************************************

import numpy as np
import h5py

import utils


def main():
   
    #tag = 'gri'
    #tag = 'gri_3signorm'
    #imtag = 'gri_lambda0.3_1.5sigdisc'
    #tag = 'gri_lambda0.3_1.5sigdisc'
    #imtag = 'gri_lambda0.3_control'
    #tag = 'gri_lambda0.3_control'
    imtag = 'gri'
    tag = 'gri_lambda0.3'

    base_dir = '/scratch/ksf293/anomalies'
    imarr_fn = f'{base_dir}/data/images_h5/images_{imtag}.h5'
    results_dir = f'{base_dir}/results'
    results_fn = f'{results_dir}/results_{tag}.h5'

    print("Loading data & residuals")
    imarr = h5py.File(imarr_fn, "r")
    res = h5py.File(results_fn, "a")

    resdict_fn = f'{base_dir}/data/idxdicts_h5/idx2resloc_{tag}.npy'
    idx2resloc = np.load(resdict_fn, allow_pickle=True).item()

    NSIDE = 96
    NBANDS = 3
    if 'residuals' in res.keys():
        del res['residuals']
    res.create_dataset('residuals', (len(res['idxs']),NSIDE,NSIDE,NBANDS), chunks=(1,NSIDE,NSIDE,NBANDS), dtype='uint8')

    for iloc in range(len(imarr['idxs'])):
        idx = imarr['idxs'][iloc]
        real = imarr['images'][iloc]
        rloc = idx2resloc[idx]
        if rloc>=len(res['idxs']):
            # idxs has an extra empty array value; make sure this doesn't break the res update
            continue
        recon = res['reconstructed'][rloc]

        real = np.array(real).reshape((NSIDE,NSIDE,NBANDS))
        real = utils.luptonize(real).astype('int')
        recon = np.array(recon).reshape((NSIDE,NSIDE,NBANDS)).astype('int')
        resid = abs(real-recon)
        res['residuals'][rloc] = resid
        
    imarr.close()
    res.close() 
    print("Done")


if __name__=='__main__':
    main()
