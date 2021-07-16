# **************************************************
# * File Name : fix_ae_anomaly_scores.py
# * Creation Date : 2021-07-09
# * Created By : kstoreyf
# * Description :
# **************************************************

import h5py


def main():

    #tag = 'gri_1k_lambda0.3'
    tag = 'gri_lambda0.3'

    aenum = 30000
    mode = 'reals'
    aetag = f'_latent64_{mode}_long_lr1e-4'
    savetag = f'_model{aenum}{aetag}'

    results_dir = f'/scratch/ksf293/anomalies/results'
    results_ae_fn = f'{results_dir}/results_ae_{tag}{savetag}.h5'
    print(results_ae_fn)

    print("Loading results")
    res = h5py.File(results_ae_fn, "a")
    print(res.keys())

    print("Computing sigma anomaly scores")
    res['ae_anomaly_scores'][...] = res['ae_anomaly_scores'][:]/255.

    res.close()
    print("Done")


if __name__=='__main__':
    main()
