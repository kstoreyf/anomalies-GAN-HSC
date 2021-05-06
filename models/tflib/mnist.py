import numpy

import os
import urllib.request, urllib.parse, urllib.error
import gzip
import pickle as pickle

def mnist_generator(images, batch_size, limit=None):
    print(images.shape)
    numpy.random.shuffle(images)
    if limit is not None:
        print("WARNING ONLY FIRST {} MNIST DIGITS".format(limit))
        images = images.astype('float32')[:limit]

    def get_epoch():
        numpy.random.shuffle(images)

        image_batches = images.reshape(-1, batch_size, 784)

        for i in range(len(image_batches)):
            yield numpy.copy(image_batches[i])

    return get_epoch

def load(batch_size, test_batch_size, n_labelled=None):
    filepath = '/tmp/mnist.pkl.gz'
    url = 'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'

    if not os.path.isfile(filepath):
        print("Couldn't find MNIST dataset in /tmp, downloading...")
        urllib.request.urlretrieve(url, filepath)

    with gzip.open('/tmp/mnist.pkl.gz', 'rb') as f:
        train_data, dev_data, test_data = pickle.load(f, encoding='latin1')
    return (
        mnist_generator(train_data[0], batch_size), 
        mnist_generator(dev_data[0], test_batch_size), 
        mnist_generator(test_data[0], test_batch_size)
    )
