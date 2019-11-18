# **************************************************
# * File Name : compute_stats.py
# * Creation Date : 2019-09-15
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np
import utils

tag = 'gri'
results_dir = '/scratch/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.h5'
imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_h5/images_{tag}.h5'

sigma = 0
reals, recons, gen_scores, disc_scores, scores, idxs, object_ids = utils.get_results(results_fn, imarr_fn, sigma=sigma)

mean = np.mean(scores)
std = np.std(scores)
print(f"Total number of objects: {len(scores)}")
n_3sig = len([s for s in scores if s>mean+3*std])
print(f"Number of 3 sigma anomalies: {n_3sig}")
n_4000 = len([s for s in scores if s>mean+4000])
print(f"Number of scores above 4000: {n_4000}")
