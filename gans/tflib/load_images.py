import numpy as np
import os



def image_generator(images, batch_size, limit=None):
    print(images.shape)
    NSIDE = images.shape[-1]
    np.random.shuffle(images)
    if limit is not None:
        print("WARNING ONLY FIRST {} IMAGES".format(limit))
        images = images.astype('float32')[:limit]

    def get_epoch():
        np.random.shuffle(images)
        
        image_batches = images.reshape(-1, batch_size, NSIDE*NSIDE)

        for i in range(len(image_batches)):
            yield np.copy(image_batches[i])

    return get_epoch


def load(batch_size, test_batch_size, imarr_fn):
    imarr = np.load(imarr_fn)
    # cut to nearest batch size below
    ntotal = int(np.floor(len(imarr)/batch_size)*batch_size)
    imarr = imarr[:ntotal]
    print(imarr.shape)
    print(np.max(imarr.flatten()), np.min(imarr.flatten()))
    #ntrain = int(int(len(imarr)/batch_size)*0.8)*batch_size
    ntrain = int(int(ntotal/batch_size)*0.8)*batch_size
    train_data = imarr[:ntrain]
    dev_data = imarr[ntrain:]
    
    return (
        image_generator(train_data, batch_size), 
        image_generator(dev_data, test_batch_size), 
    )
