#!/usr/bin/env python3
import sys
import glob
import os
import numpy as np
from matplotlib import pyplot as plt

def main(dirname):
    glob_pattern = os.path.join(dirname,"rbm_xsample_epoch*npy")
    filenames = glob.glob(glob_pattern)
    filenames.sort()
    for filename in filenames:
        print(filename)
        xsample = np.load(filename)
        im = xsample[1:,:].reshape(28,28)
        plt.imshow(im,interpolation='none')
        plt.show()

if __name__=="__main__":
    main(sys.argv[1])
