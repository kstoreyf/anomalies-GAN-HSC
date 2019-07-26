import numpy as np


def main():
    #tag = 'i60k_28x28'
    tag = 'i60k_96x96'
    imarr_fn = "imarrs_np/hsc_{}.npy".format(tag)
    data = np.load(imarr_fn)

    normed = [normalize(d) for d in data]
    np.save("imarrs_np/hsc_{}_norm.npy".format(tag), normed)

def norm0to1(a):
    return (a - np.min(a))/np.ptp(a)

def normalize(d):
    d = np.arcsinh(d)
    d = norm0to1(d)
    return(d)

if __name__=='__main__':
    main()
