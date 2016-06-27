#!/usr/bin/env python

import numpy as np
import random
import sys
from matplotlib import pyplot as plt

mu = None
sigma = None

def sample_Q(x):
    return x + random.uniform(-1,1)

def unnormalized_normal(x):
    return np.exp(
        - (
            (x - mu) ** 2
        ) / ( 2 * sigma)
    )

def sample():
    x = 0.
    while True:
        yield x
        x_next = sample_Q(x)
        u = random.uniform(0,1)
        pi_tilde_x_next = unnormalized_normal(x_next)
        pi_tilde_x = unnormalized_normal(x)
        pi_ratio = pi_tilde_x_next / pi_tilde_x
        if u < pi_ratio:
            # accept
            x = x_next
        else:
            # reject
            pass # no action taken, will return copy of previous sample


def many_samples(howmany=1000):
    generator = sample()
    xs = []
    for i in xrange(howmany):
        xs.append(generator.next())
    return xs

def plot(xs):
    plt.hist(xs,bins=200)
    plt.show()

def main():
    global mu
    mu = float(sys.argv[1])
    global sigma
    sigma = float(sys.argv[2])
    xs = many_samples(howmany=int(sys.argv[3]))
    plot(xs)

if __name__=="__main__":
    main()

