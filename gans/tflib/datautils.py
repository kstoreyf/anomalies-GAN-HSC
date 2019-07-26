#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Data utils
Created on Tue Jul 23 10:40:26 2019
@author: ribli

Modified by kstoreyf
"""

import numpy as np


class DataGenerator():
    """
    Data generator.
    Generates minibatches of data and labels.
    Usage:
    from imgen import ImageGenerator
    g = DataGenerator(data, labels)
    """
    def __init__(self, x, y=None, batch_size=32, shuffle=True, seed=0,
                 augment = False):
        """Initialize data generator."""
        print(x.shape)
        self.x, self.y = x, y
        if self.y == None:  # if no labels make up fake labels
            self.y = 42 * np.ones(len(self.x))
        assert self.x.shape[1] == self.x.shape[2]  # square! for augmentation
        self.x_shape = x.shape[1] * x.shape[2]  # return flat images for now
        
        self.batch_size = batch_size           
        self.shuffle = shuffle
        self.augment = augment
        self.seed = seed
        self.rng = np.random.RandomState(self.seed)

        self.n_data = len(x)
        self.n_steps = len(x)//batch_size  +  (len(x) % batch_size > 0)
        self.i = 0
        self.reset_indices_and_reshuffle(force=True)
        
        
    def reset_indices_and_reshuffle(self, force=False):
        """Reset indices and reshuffle images when needed."""
        if self.i == self.n_data or force:
            if self.shuffle:
                self.index = self.rng.permutation(self.n_data)
            else:
                self.index = np.arange(self.n_data)
            self.i = 0
            
                
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
        x = self.x[self.index[self.i]].copy()
        y = self.y[self.index[self.i]]
        x = self.process_image(x)      
        self.i += 1  # increment counter
        return x, y
    
    
    def process_image(self, x):
        """Process data."""   
        if self.augment:  # flip and transpose
            # this is not correct now, labels change too!!!
            x = aug_im(x, self.rng.rand()>0.5, self.rng.rand()>0.5, 
                       self.rng.rand()>0.5)  
        x = x.reshape(self.x_shape)
        return  x


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
