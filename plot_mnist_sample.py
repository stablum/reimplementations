import numpy as np
import sys
from matplotlib import pyplot as plt

def main():
    rasters = []
    for filename in sys.argv[1:]:
        a = np.load(filename)
        print("a.shape",a.shape)
        for curr in a[0,:,:10].T:
            print("curr.shape",curr.shape)
            try:
                raster = np.reshape(a,(28,28))
            except Exception as e:
                print ("curr.shape",curr.shape)
                raise e
            raster = np.reshape(curr,(28,28))
            rasters.append(raster)
    rasters_stacked = np.hstack(rasters)
    plt.imshow(rasters_stacked)
    plt.show()

if __name__=="__main__":
    main()
