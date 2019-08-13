#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data utils
Created on Tue Jul 23 10:40:26 2019
@author: ribli

Modified by kstoreyf
"""
import time
import numpy as np
import h5py
import cv2

from astropy.visualization import make_lupton_rgb
from scipy.ndimage import gaussian_filter


class DataGenerator():
    """
    Data generator.
    Generates minibatches of data and labels.
    Usage:
    from imgen import ImageGenerator
    g = DataGenerator(data, labels)
    """
    def __init__(self, x, y=None, batch_size=32, shuffle=True, seed=0,
                 augment = False, luptonize=False, normalize=False, 
                 shuffle_chunk=1000, smooth=True):
        """Initialize data generator."""
        print(x.shape)
        self.x, self.y = x, y
        if self.y == None:  # if no labels make up fake labels
            self.y = 42 * np.ones(len(self.x))
        assert self.x.shape[1] == self.x.shape[2]  # square! for augmentation
        assert not (luptonize and normalize), "Can't have both luptonize and normalize"
        #self.x_shape = x.shape[1] * x.shape[2]  # return flat images for now
        if len(x.shape)>3:
            self.x_shape = x.shape[1] * x.shape[2] * x.shape[3]
        else:
            self.x_shape = x.shape[1] * x.shape[2]

        self.batch_size = batch_size           
        self.shuffle = shuffle
        self.augment = augment
        self.luptonize = luptonize
        self.normalize = normalize
        self.shuffle_chunk = shuffle_chunk
        self.smooth = smooth
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.n_data = len(x)
        self.n_steps = len(x)//batch_size  +  (len(x) % batch_size > 0)
        self.i = 0
        self.reset_indices_and_reshuffle(force=True)
        
        
        
    def reset_indices_and_reshuffle(self, force=False):
        """Reset indices and reshuffle images when needed."""
        if self.i == self.n_data or force:
            print("shuffling")
            if self.shuffle:
                if self.shuffle_chunk:
                    self.index = self.get_indices()
                else:
                    self.index = self.rng.permutation(self.n_data)
            else:
                self.index = np.arange(self.n_data)
            self.i = 0
            

    def get_indices(self):
        nchunks = int(self.n_data/self.shuffle_chunk)
        if self.n_data % self.shuffle_chunk != 0:
            nchunks += 1
        starts = np.random.choice(nchunks, size=nchunks, replace=False)
        indices = []
        for s in starts:
            for i in range(self.shuffle_chunk):
                idx = s*self.shuffle_chunk+i
                if idx<self.n_data:
                    indices.append(idx)
        return indices
                
    def next(self):
        """Get next batch of images."""
        x = np.zeros((self.batch_size, self.x_shape))
        y = np.zeros((self.batch_size,))
        for i in range(self.batch_size):
            x[i], y[i] = self.next_one()        
        return x,y
    
    
    def next_one(self):
        """Get next 1 image."""
        # reset index, reshuffle if necessary
        self.reset_indices_and_reshuffle()  
        # get next x
        #x = self.x[self.index[self.i]].copy()
        idx = self.index[self.i]
        #ss = time.time()
        x = self.x[idx].copy()
        #ee = time.time()
        #print(self.i)
        #print(idx)
        #print(f"t {ee-ss}") 
        y = self.y[idx]
        x = self.process_image(x)      
        self.i += 1  # increment counter
        return x, y
   
    def sample(self, n):
        """Get next 1 image."""
        # reset index, reshuffle if necessary
        #self.reset_indices_and_reshuffle()
        x = np.zeros((n, self.x_shape))
        y = np.zeros((n,))
        # get next x
        for i in range(n):
            xi = self.x[i].copy()
            x[i] = self.process_image(xi)
            y[i] = self.y[i]
        return x, y
    
    def process_image(self, x):
        """Process data."""   
        if self.augment:  # flip and transpose
            # this is not correct now, labels change too!!!
            x = aug_im(x, self.rng.rand()>0.5, self.rng.rand()>0.5, 
                       self.rng.rand()>0.5)  
        if self.luptonize:
            rgb_q = 15
            rgb_stretch = 0.5
            rgb_min = 0
            #x = make_lupton_rgb(x[:,:,2], x[:,:,1], x[:,:,0],
            #             Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
            x = make_lupton_rgb(x[:,:,2], x[:,:,2], x[:,:,2],
                                     Q=rgb_q, stretch=rgb_stretch, minimum=rgb_min)
            x = np.array(x)
            x = np.array([xi/255. for xi in x])
        if self.smooth:
            #x = np.array([gaussian_filter(xi, sigma=1) for xi in x])
            x = cv2.blur(x,(5,5))
        if self.normalize:
            x = np.arcsinh(x)
            x = (x - np.min(x))/np.ptp(x)
        x = x.reshape(self.x_shape)
        return x


def aug_im(im, fliplr=0, flipud=0, T=0):
    """Augment images with flips and transposition."""
    if fliplr:  # flip left right
        im = np.fliplr(im)
    if flipud:  # flip up down
        im = np.flipud(im)
    if T:  # transpose
        #for i in xrange(im.shape[-1]):
        #    im[:,:,i] = im[:,:,i].T
        im = im.T
    return im


def load_numpy(fn, split=False):
    arrs = np.load(fn)
    if split:
        ntrain = int(0.8*len(arrs))
        train = arrs[:ntrain]
        test = arrs[ntrain:]
        return train, test
    else:
        return arrs

def load(fn, dataset='images'):
    if fn.endswith(".npy"):
        return load_numpy(fn)
    elif fn.endswith(".h5"):
        f = h5py.File(fn,"r")
        return f[dataset]
        #return np.array(f[dataset])
