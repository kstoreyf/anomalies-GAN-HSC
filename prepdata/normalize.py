import numpy as np


def main():
    #tag = 'i60k_28x28'
    tag = 'i20.0'
    imarr_fn = "/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{}.npy".format(tag)
    #imarr_fn = "/home/ksf293/kavli/anomalies-GAN-HSC/data/imarrs_np/hsc_{}.npy".format(tag)
    data = np.load(imarr_fn)
    print(len(data))
    normed = [normalize(d) for d in data]
    np.save("/scratch/ksf293/kavli/anomaly/data/images_np/imarr_{}_norm.npy".format(tag), normed)

def norm0to1(a):
    return (a - np.min(a))/np.ptp(a)

def normalize(d):
    d = np.arcsinh(d)
    d = norm0to1(d)
    return(d)

if __name__=='__main__':
    main()
