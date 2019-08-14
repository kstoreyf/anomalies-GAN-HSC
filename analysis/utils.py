# **************************************************
# * File Name : utils.py
# * Creation Date : 2019-08-14
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np
from astropy.visualization import make_lupton_rgb
import h5py

NBANDS = 3

def luptonize(x):
    rgb_q = 15
    rgb_stretch = 0.5
    rgb_min = 0
    if x.ndim==3:
        x = make_lupton_rgb(x[:,:,2], x[:,:,1], x[:,:,0],
                      Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
    elif x.ndim==4:
        x = np.array([make_lupton_rgb(xi[:,:,2], xi[:,:,1], xi[:,:,0],
                      Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
                      for xi in x])
    else:
        raise ValueError(f"Wrong number of dimensions! Gave {x.ndim}, need 3 or 4")
    return x

def get_results(result_fn, imarr_fn, n_anoms=0, sigma=0):
    print("Loading results")    
    res = h5py.File(result_fn)
    imarr = h5py.File(imarr_fn)
    
    if sigma:
        scores = res['anomaly_scores']
        mean = np.mean(scores)
        std = np.std(scores)
        n_anoms = len([s for s in scores if s>mean+sigma*std])
        print(f"Number of {sigma}-sigma anomalies: {n_anoms}")
        if n_anoms==0:
            raise ValueError(f"No {sigma}-sigma anomalies")

    if n_anoms>0:
        idx_sorted = np.argsort(res['anomaly_scores'])
        sample = list(idx_sorted[-n_anoms:])
        reals = [imarr['images'][s] for s in sample]
        recons = [res['reconstructed'][s] for s in sample]
        gen_scores = [res['gen_scores'][s] for s in sample]
        disc_scores = [res['disc_scores'][s] for s in sample]
        scores = [res['anomaly_scores'][s] for s in sample]
        idxs = [res['idxs'][s] for s in sample] 
        object_ids = [res['object_ids'][s] for s in sample]
    else:
        reals = imarr['images']
        recons = res['reconstructed']
        gen_scores = res['gen_scores']
        disc_scores = res['disc_scores']
        scores = res['anomaly_scores']
        idxs = res['idxs']
        object_ids = res['object_ids']
    return reals, recons, gen_scores, disc_scores, scores, idxs, object_ids    

def get_residuals(reals, recons):
    print("Getting residuals")
    reals = np.array(reals)
    reals = reals.reshape((-1,96,96,NBANDS))
    reals = luptonize(reals).astype('int')
    recons = np.array(recons)
    recons = recons.reshape((-1,96,96,NBANDS)).astype('int')
    resids = abs(reals-recons)
    return resids, reals, recons
