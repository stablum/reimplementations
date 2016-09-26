import numpy as np
import sys
from matplotlib import pyplot as plt

def main():
    rasters = []
    for filename in sys.argv[1:]:
        a = np.load(filename)
        try:
            raster = np.reshape(a,(28,28))
        except Exception as e:
            print ("a.shape",a.shape)
            raise e

        rasters.append(raster)
    rasters_stacked = np.hstack(rasters)
    plt.imshow(rasters_stacked)
    plt.show()

if __name__=="__main__":
    main()
