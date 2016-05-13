#!/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
def sample_circle():
    r_sq = 2
    while r_sq > 1:
        z = np.random.uniform(-1,1,2)
        r_sq = np.sum(np.power(z,2))
    return z,r_sq

def sample_normal():
    z, r_sq = sample_circle()
    x = z * np.sqrt(-2 * np.log(r_sq) / r_sq )
    return x

def sample_many_normal(amount=100):
    ret = np.array([])
    for i in range(amount/2):
        x = sample_normal()
        ret = np.hstack([ret, x])
    return ret

def plot_sample_normal():
    xs = sample_many_normal(amount=50000)
    plt.hist(xs,bins=200)
    plt.show()

def sample_multivariate_normal():
    cov = np.array([[1,4],[4,19]])
    L = la.cholesky(cov,lower=True)
    assert la.norm(np.dot(L,L.T) - cov) == 0, "L calculated erroneously"
    x = sample_normal()
    mu = np.array([40,40])
    y = np.dot(L,x) + mu
    return y

def sample_many_multivariate_normal(amount=100):
    l = []
    for i in range(amount/2):
        x = sample_multivariate_normal()
        l.append(x)
    ret = np.array(l)
    return ret

def plot_multivariate_normal():
    xs = sample_many_multivariate_normal(amount=100000)
    plt.hist2d(xs[:,0],xs[:,1],bins=[400,400])
    plt.show()

def main():
    plot_sample_normal()
    plot_multivariate_normal()

if __name__ == "__main__":
    main()
