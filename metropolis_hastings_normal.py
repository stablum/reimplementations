#!/usr/bin/env python

import numpy as np
import random
import sys
from matplotlib import pyplot as plt

def sample_Q(x):
    return x + random.uniform(-1,1)

def unnormalized_normal(x):
    sigma = 10
    mu = 40
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
            pass # no action taken


def many_samples(howmany=1000):
    generator = sample()
    xs = []
    for i in xrange(howmany):
        xs.append(generator.next())
    return xs

def plot(xs):
    plt.hist(xs,bins=200)
    plt.show()

def main(howmany=1000):
    xs = many_samples(howmany=howmany)
    plot(xs)

if __name__=="__main__":
    main(howmany=int(sys.argv[1]))
