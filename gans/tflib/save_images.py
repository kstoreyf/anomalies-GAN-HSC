"""
Image grid saver, based on color_grid_vis from github.com/Newmu
"""

import numpy as np
import scipy.misc
#from scipy.misc import imsave
from imageio import imwrite
import matplotlib.pyplot as plt

def save_images(X, save_path):
    # [0, 1] -> [0,255]
    print(np.max(X.flatten()), np.min(X.flatten()))
    #if isinstance(X.flatten()[0], np.floating):
    #    X = (255.99*X).astype('uint8')

    n_samples = X.shape[0]
    rows = int(np.sqrt(n_samples))
    while n_samples % rows != 0:
        rows -= 1

    nh, nw = rows, n_samples/rows

    if X.ndim == 2:
        X = np.reshape(X, (X.shape[0], int(np.sqrt(X.shape[1])), int(np.sqrt(X.shape[1]))))

    if X.ndim == 4:
        # BCHW -> BHWC
        X = X.transpose(0,2,3,1)
        h, w = X[0].shape[:2]
        img = np.zeros((h*nh, w*nw, 3))
    elif X.ndim == 3:
        h, w = X[0].shape[:2]
        img = np.zeros((int(h*nh), int(w*nw)))

    for n, x in enumerate(X):
        j = n//nw
        i = n%nw
        img[int(j*h):int(j*h+h), int(i*w):int(i*w+w)] = x
    print(np.max(img.flatten()), np.min(img.flatten()))
    imwrite(save_path, img)
    #plt.figure()
    #ax = plt.gca()
    #ax.imshow(img, origin='lower', cmap='gray')
    #ax.set_xticks([])
    #ax.set_yticks([]) 
    #plt.savefig(save_path)

