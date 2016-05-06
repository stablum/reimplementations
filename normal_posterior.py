#!/usr/bin/env python

import numpy as np
import sys

def main(mu_truth=12345., sigma_truth=0.1, size=100):

    X = np.random.normal(loc=mu_truth, scale=sigma_truth, size=size)

    sigma_0 = 100.
    mu_0 = 0.
    sigma = sigma_truth

    for i,x in enumerate(list(X)):
        sigma_0 = sigma * sigma_0  / np.sqrt( (sigma ** 2) + ( sigma_0 ** 2) )
        mu_0 = (
                x * ( sigma_0 ** 2 ) + mu_0 * (sigma ** 2)
            ) / (
                ( sigma ** 2 ) + ( sigma_0 ** 2 )
            )
        print "i:",i,"x:",x,"sigma_0:",sigma_0,"mu_0:",mu_0

if __name__=="__main__":
    main(
        mu_truth=float(sys.argv[1]),
        sigma_truth=float(sys.argv[2]),
        size=float(sys.argv[3])
    )
