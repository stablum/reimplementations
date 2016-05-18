#!/usr/bin/env python
import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_dimensionality = 2
num_epochs = 10
eta = None
noise_level = 0.000001#0.1

amount_points = 250
def true_f(x):
    _theta = np.array([
        0.51, #constant
        0., #x0
        0.,#-0.2, #x1
        0,#1.5, #x0x1
    ])
    #import ipdb; ipdb.set_trace()
    tmp = np.column_stack([
        np.ones(x.shape[0]),
        x[:,0],
        x[:,1],
        x[:,0]*x[:,1]
    ])
    print tmp.shape
    tmp2 = tmp*_theta
    ret = np.sum(tmp2,1)
    return ret

def sigmoid(t):
    return 1./( 1. + np.exp(-t))

def augment(x):
    if len(x.shape) == 2:
        oo = np.ones(x.shape[0])
        augmented_x = np.column_stack([oo,x])
    else:
        augmented_x = np.hstack([1,x])
    return augmented_x

def the_dot(ws,x):
    augmented_x = augment(x)
    a = np.dot(augmented_x,ws)
    return a

def f(ws, x):
    #print "x.shape",x.shape
    a = the_dot(ws, x)
    return sigmoid(a)

def generate_dataset():
    xs = np.random.normal(-1,1,(amount_points,2))
    noise = np.random.normal(0,noise_level,amount_points)
    ys = true_f(xs) + noise
    return xs,ys

def plot_data(xs,ys):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(xs[:,0],xs[:,1],ys)
    plt.show()

def loss(ws,xs,ys):
    yhats = f(ws,xs)
    ret = np.sum((ys - yhats)**2)
    return ret

def grad_loss(ws,xi,yi):
    predicted = f(ws,xi)
    delta = predicted - yi
    a = the_dot(ws,xi)
    sigmoid_derivative = sigmoid(a) * ( 1. - sigmoid(a) )

    #print "delta",delta, "predicted",predicted,"a",a,"sigmoid_derivative",sigmoid_derivative
    if np.isnan(sigmoid_derivative):
        print "a",a
        print "ws",ws
        print "xi",xi
        import ipdb; ipdb.set_trace()
    common = 2 * sigmoid_derivative * delta
    ret = common * augment(xi) # it's the vector of individual derivatives
    return ret

def sgd_estimation(ws_init,xs,ys):
    global num_epochs
    global eta
    ws = ws_init
    for epoch in range(num_epochs):
        for i,(xi,yi) in enumerate(zip(xs,ys)):
            _grad = grad_loss(ws, xi, yi)
            ws = ws - eta * _grad
            #print "xi",xi
            #print "yi",yi
            #print "_grad",_grad
            #print "ws",ws
        print epoch,"loss",loss(ws, xs, ys),"ws",ws

def main():
    global eta
    global num_epochs

    np.set_printoptions(precision=6, suppress=True)

    eta = float(sys.argv[1])
    num_epochs = int(sys.argv[2])

    xs,ys = generate_dataset()
    #plot_data(xs,ys)

    # input_dimensionality+1 because there is also the bias weight
    ws = np.random.normal(-1,1,input_dimensionality+1)
    print "ws",ws
    yhats = np.array([
        f(ws, _x)
        for _x
        in xs
    ])
    #plot_data(xs, yhats)
    print "loss",loss(ws,xs,ys)
    new_w = sgd_estimation(ws,xs,ys)

if __name__=="__main__":
    main()
