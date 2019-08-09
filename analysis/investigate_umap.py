# **************************************************
# * File Name : investigate_umap.py
# * Creation Date : 2019-08-05
# * Created By : kstoreyf
# * Description :
# **************************************************
import numpy as np

import save_images as saver

tag = 'i20.0_norm_100k_features0.05go'
savetag = '_auto'
#mode = 'images'
mode = 'residuals'
emb_fn = f'/scratch/ksf293/kavli/anomaly/results/embedding_{tag}{savetag}.npy'
embed = np.load(emb_fn, allow_pickle=True)

#imarr_fn = f'/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{tag}.npy'
#imarr = np.load(imarr_fn)
results_dir = '/scratch/ksf293/kavli/anomaly/results'
results_fn = f'{results_dir}/results_{tag}.npy'

plot_dir = '/home/ksf293/kavli/anomalies-GAN-HSC/plots/plots_2019-08-08'

res = np.load(results_fn, allow_pickle=True)
images = res[:,0]
images = np.array([np.array(im) for im in images])
images = images.reshape((-1, 96*96))

if mode=='images':
    imarr = images
    modetag = '-orig'
elif mode=='residuals':
    recons = res[:,1]
    recons = np.array([np.array(im) for im in recons])
    recons = recons.reshape((-1, 96*96))    
    imarr = abs(images-recons)
    modetag = ''
    #imarr = images-recons
    #cmap = 'coolwarm'

masks = [(embed[0]>20), ((-3<embed[0]) & (embed[0]<2) & (-2<embed[1]) & (embed[1]<3))]
masktags = ['right', 'center']
#masks = [(embed[1]>5.5), ((embed[0]>5.5) & (embed[1]<-2)),
#         ((embed[0]<-4) & (embed[1]<-3))]
#masktags = ['top', 'lowerright', 'lowerleft']

def make_batch(arr, save_fn, n=128):
    ims = arr[:n]
    ims.reshape((-1, 96, 96))
    saver.save_images(ims, save_fn)
#leftidxs = [embed[3][i] for i in range(len(embed[0])) if embed[0][i]<6]
#rightidxs = [embed[3][i] for i in range(len(embed[0])) if embed[0][i]>=6]
#print(leftidxs)
#print(rightidxs)
for i in range(len(masks)):
    idxs = embed[3][masks[i]]
    #idxs = [embed[3][i] for i in range(len(embed[0])) if masks[i]]
    ims = np.array([imarr[idx] for idx in idxs])
    make_batch(ims, f'{plot_dir}/cluster_{tag}{savetag}{modetag}_{masktags[i]}.png')

#leftims = np.array([imarr[idx] for idx in leftidxs])
#rightims = np.array([imarr[idx] for idx in rightidxs])
#print(len(leftims))
#print(len(rightims))

#make_batch(leftims, f'{plot_dir}/cluster_{tag}{savetag}_left.png')
#make_batch(rightims, f'{plot_dir}/cluster_{tag}{savetag}_right.png')
