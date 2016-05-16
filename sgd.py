#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt

true_theta = None
num_epochs = 10
eta = None
max_degree = 3
def f(xs,_theta):
    ret = _theta[0] + (_theta[1] * xs) + (_theta[2] * (xs ** 2)) + (_theta[3] * (xs ** 3))
    return ret

def generate_dataset():
    global true_theta
    amount_points = 250
    xs = np.random.normal(0,2,amount_points)
    noise = np.random.normal(0,0.1,amount_points)
    ys = f(xs,true_theta) + noise
    return xs,ys

def plot_data(xs,ys):
    plt.scatter(xs,ys)
    plt.show()

def loss(_theta,xs,ys):
    yhats = f(xs,_theta)
    ret = np.sum((ys - yhats)**2)
    return ret

def grad_loss(_theta,xi,yi):
    global max_degree
    f_xi = f(xi,_theta)

    return 2 * ( (- yi) + f_xi) * np.array([
        (xi ** degree)
        for degree
        in range(max_degree+1)
    ])

def sgd_estimation(xs,ys):
    global num_epochs
    global eta
    theta = np.random.uniform(-1,1,max_degree+1)
    for epoch in range(num_epochs):
        for i,(xi,yi) in enumerate(zip(xs,ys)):
            theta = theta - eta * grad_loss(theta, xi, yi)
        print epoch,"loss",loss(theta, xs, ys),"theta",theta

def main():
    global true_theta
    global eta
    global num_epochs

    np.set_printoptions(precision=6, suppress=True)

    true_theta = np.zeros(max_degree+1)
    eta = float(sys.argv[1])
    num_epochs = int(sys.argv[2])
    for i in range(max_degree+1):
        true_theta[i] = float(sys.argv[3+i])

    xs,ys = generate_dataset()
    plot_data(xs,ys)

    new_theta = sgd_estimation(xs,ys)

if __name__=="__main__":
    main()
