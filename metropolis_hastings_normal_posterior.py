#!/usr/bin/env python

import numpy as np
import random
import sys
from matplotlib import pyplot as plt

sigma = None
mu0 = None
sigma0 = None

def sample_Q(x):
    return x + random.uniform(-1,1)

def unnormalized_log_likelihood(data, mu):
    global sigma

    _sum = np.sum([(x - mu) ** 2 for x in data])
    return - (
        _sum
    ) / ( 2 * sigma)

def log_prior(mu):
    global mu0
    global sigma0
    return unnormalized_log_normal(mu, mu0, sigma0)

def unnormalized_log_posterior(data, mu):
    l = unnormalized_log_likelihood(data, mu)
    p = log_prior(mu)
    return l + p

def unnormalized_log_normal(x,_mu,_sigma):
    return - (
        (x - _mu) ** 2
    ) / ( 2 * _sigma)

def sample(data):
    mu = 0.
    while True:
        yield mu
        mu_next = sample_Q(mu)
        u = random.uniform(0,1)
        pi_tilde_mu_next = unnormalized_log_posterior(data, mu_next)
        pi_tilde_mu = unnormalized_log_posterior(data, mu)
        pi_ratio = pi_tilde_mu_next - pi_tilde_mu
        if np.log(u) < pi_ratio:
            # accept
            mu = mu_next
        else:
            # reject
            pass # no action taken, will return copy of previous sample

def generate_data(dataset_size):
    global true_mu
    global sigma
    data = np.random.normal(true_mu, sigma, dataset_size)
    return data

def many_samples(howmany=1000,dataset_size=1000):
    data = generate_data(dataset_size)
    generator = sample(data)
    mus = []
    for i in xrange(howmany):
        mus.append(generator.next())
    return mus

def plot(xs):
    plt.hist(xs,bins=200)
    plt.show()

def main():
    global true_mu
    true_mu = float(sys.argv[1])
    global sigma
    sigma = float(sys.argv[2])
    global mu0
    mu0 = float(sys.argv[3])
    global sigma0
    sigma0 = float(sys.argv[4])
    mus = many_samples(howmany=int(sys.argv[6]),dataset_size=int(sys.argv[5]))
    plot(mus)
    print "average mus:",np.mean(mus)
    print "average mus (second half):",np.mean(mus[len(mus)/2 :])

if __name__=="__main__":
    main()
