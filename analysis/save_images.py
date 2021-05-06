# ********************************************************
# * File Name : save_images.py
# * Creation Date : 
# * Created By : kstoreyf
# * Description : Image grid saver, based on 
#                 color_grid_vis from github.com/Newmu
# ********************************************************
import sys
import numpy as np
import scipy.misc
from imageio import imwrite
import matplotlib.pyplot as plt
from astropy.visualization import make_lupton_rgb

np.set_printoptions(threshold=sys.maxsize)

def save_images(X, save_path, luptonize=False, unnormalize=False):
    # [0, 1] -> [0,255]
    if luptonize:
        unnormalize = False 
    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        if unnormalize:
            X = (255.*X).astype('uint8')
        h, w = X[0].shape[:2]
        img = np.zeros((int(h*nh), int(w*nw), 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((int(h*nh), int(w*nw)))
    
    rgb_q = 15
    rgb_stretch = 0.5
    rgb_min = 0
    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        if luptonize:
            x = make_lupton_rgb(x[:,:,2], x[:,:,1], x[:,:,0],
                         Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)    
        if n==0:
            print(x.shape)
            print(x[40:-40,40:-40])
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = x
    imwrite(save_path, img)
 
