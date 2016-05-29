#!/usr/bin/env python

import numpy as np
import scipy.stats
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def plot_orig(mu_orig, sigma_orig):
    pdf = lambda x,y : scipy.stats.multivariate_normal.pdf(np.array([x,y]), mean=mu_orig, cov=sigma_orig)
    plot_pdf(pdf)

def plot_variational(m1,s1,m2,s2):
    def pdf(x, y):
        p1 = scipy.stats.norm.pdf(x,m1,s1)
        p2 = scipy.stats.norm.pdf(y,m2,s2)
        #print "p1",p1,m1,s1
        #print "p2",p2
        ret = p1*p2
        #print "ret",ret
        return ret
    plot_pdf(pdf)

def plot_pdf(pdf):
    raster = np.zeros((100,100))
    for x in xrange(100):
        for y in xrange(100):
            p = pdf(x-50,y-50)
            #print p
            raster[y,x] = p
    plt.imshow(raster)
    plt.colorbar()
    plt.show()

def main():
    mu_orig = np.array([40.,-20.])
    mu1,mu2 = mu_orig
    sigma_orig = np.array([[5.,5.],[5.,10.]])
    ((s11,s12),(s21,s22)) = sigma_orig
    si = np.linalg.inv(sigma_orig)
    ((si11,si12),(si21,si22)) = si
    print "si",si
    n_iter = 20
    mu_variational_1 = 0.
    mu_variational_2 = 0.
    m1 = 0.
    m2 = 0.
    sigma_variational_1 = 1 / si11
    sigma_variational_2 = 1 / si22
    for i in range(n_iter):
        mu_variational_1 = mu1 + 0.5 \
            * (mu2 - mu_variational_2) \
            * (si12+si21)/si11
        mu_variational_2 = mu2 + 0.5 \
            * (mu1 - mu_variational_1) \
            * (si12+si21)/si22

        print "mu_variational",mu_variational_1,mu_variational_2

    print "sigma_variational",sigma_variational_1,sigma_variational_2
    plot_orig(mu_orig,sigma_orig)
    plot_variational(mu_variational_1, sigma_variational_1, mu_variational_2, sigma_variational_2)


if __name__=="__main__":
    main()
