#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
def pdf(x, _lambda=1):
    assert all([ curr >= 0 for curr in x]),"only positive values for x"
    return _lambda * np.exp(-_lambda * x)

def plot_pdf(_lambda=1):
    x = np.linspace(0,5,10)
    probs = pdf(x,_lambda=_lambda)
    plt.plot(x,probs)
    plt.show()

def cdf(x, _lambda=1):
    return 1 - np.exp(-_lambda * x)

def plot_cdf(_lambda=1):
    x = np.linspace(0,5,10)
    F_vals = cdf(x,_lambda=_lambda)
    plt.plot(x,F_vals)
    plt.show()

def cdf_inverse(probs,_lambda=1):
    return - (np.log(1-probs)/_lambda)

def plot_cdf_inverse(_lambda=1):
    probs = np.linspace(0,1,10)
    F_inv_vals = cdf_inverse(probs,_lambda=_lambda)
    plt.plot(probs,F_inv_vals)
    plt.show()

def sample(_lambda=1,amount=10):
    u = np.random.uniform(0,1,amount)
    samples = cdf_inverse(u,_lambda=_lambda)
    return samples

def plot_sample():
    samples = sample(_lambda=1,amount=10000)
    plt.hist(samples,bins=100)
    plt.show()

def main():
    #plot_pdf()
    #plot_cdf()
    #plot_cdf_inverse()
    plot_sample()

if __name__=="__main__":
    main()
