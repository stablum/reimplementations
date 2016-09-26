#!/usr/bin/env python3

import theano
from theano import tensor as T
import pandas as pd
from tqdm import trange,tqdm
import numpy as np
from sklearn.preprocessing import normalize
import sklearn.svm
import time
import mnist # pip3 install python-mnist
import os
import sys

#theano.config.exception_verbosity="high"
#theano.config.optimizer='None'

theano.config.optimizer='fast_run'
theano.config.openmp=False
theano.config.openmp_elemwise_minsize=10
#theano.config.device='gpu'
theano.config.floatX='float32'
lr_begin = 0.02 # 0.2
lr_annealing_T=10
lr=None
n_epochs = 10000
data_amplify = 0.5
data_offset = 0.25
sigma_x = 0.5
sigma_z = 1#1e2#1e4
latent_dim = None
hidden_dims = None
num_hidden_layers = None
activation_function = None
#activation_function_inverse = lambda x: T.log(x) - T.log(1-x) # logit
minibatch_size = None
repeat_training=1

possible_activations = {
    'sigmoid': T.nnet.sigmoid,

    # 2.37 seems to make a sigmoid a good approximation for erf(x),
    'pseudogelu': lambda x: x * T.nnet.sigmoid(x*2.37),

    'gelu': lambda x : x*T.erf(x),
    'elu': T.nnet.elu,
    'relu': T.nnet.relu
}

class Logger():
    def __init__(self):
        self.filename = "theano_decoder_"+str(time.time())+".log"
        self.f = open(self.filename,'w')

    def __call__(self, *args):
        print(*args, flush=True)
        print(*args,file=self.f, flush=True)

log = None

def calculate_lr(t):
    # decaying learning rate with annealing
    # see: https://www.willamette.edu/~gorr/classes/cs449/momrate.html
    ret = lr_begin / (
        1. + float(t)/lr_annealing_T
    )
    return ret

def shuffle(Z,X,Y):
    sel = np.arange(X.shape[1])
    np.random.shuffle(sel)
    X = X[:,sel]
    Z = Z[:,sel]
    Y = Y[:,sel]
    return Z,X,Y

def fix_data(features,labels):
    # please notice the transpose '.T' operator
    # in a neural network, the datapoints needs to be scattered across the columns
    # because dot product.
    X = (np.array(features).T.astype('float32')/255.)*data_amplify + data_offset
    Y = np.expand_dims(np.array(labels).astype('float32'),1).T
    return X,Y

def load_data():
    print("setting up mnist loader..")
    _mnist = mnist.MNIST(path='./python-mnist/data')
    print("loading training data..")
    X_train,Y_train = fix_data(*_mnist.load_training())
    print("X_train.shape=",X_train.shape,"Y_train.shape=",Y_train.shape)
    print("loading testing data..")
    X_test,Y_test = fix_data(*_mnist.load_testing())
    print("X_test.shape=",X_test.shape,"Y_test.shape=",Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def add_layer(inputs,i):
    assert type(i) is int
    W = T.matrix('W%d'%i)
    bias = T.matrix('bias%d'%i)
    bias_tiled = bias.repeat(minibatch_size, axis=1)
    d = T.dot(W, inputs)
    d.name = "d%d"%i
    lin = d + bias_tiled
    lin.name = 'lin%d'%i
    out = activation_function(lin)
    out.name = 'out%d'%i
    return W,bias,out

def add_layer_inverse(curr, i, W, bias):
    assert type(i) is int
    lin = activation_function_inverse(curr)
    lin.name = 'lin_inverse%d'%i
    bias_tiled = bias.repeat(minibatch_size, axis=1)
    bias_tiled.name = 'bias_tiled_inverse%d'%i
    d = lin - bias_tiled
    d.name = "d_inverse%d"%i
    W_pinv = T.nlinalg.MatrixPinv()(W)
    W_pinv.name = "W_pinv%d"%i
    ret = T.dot(W_pinv,d)
    return ret

def build_net_inverse(xs,Ws,biases):
    curr = xs
    for i, (W, bias) in enumerate(reversed(list(zip(Ws,biases)))):
        curr = add_layer_inverse(curr, i, W, bias)
    return curr

def build_net():
    curr = inputs = T.matrix('inputs')
    Ws = []
    biases = []
    for i in range(num_hidden_layers + 1):
        W, bias, curr = add_layer(curr,i)
        Ws.append(W)
        biases.append(bias)

    return inputs, Ws, biases, curr

def update(learnable, grad):
    learnable -= lr * grad

def step(zs, xs, Ws_vals, biases_vals, grad_fn):
    grad_vals = grad_fn(*([zs, xs] + Ws_vals + biases_vals))
    Ws_grads = grad_vals[:len(Ws_vals)]
    biases_grads = grad_vals[len(Ws_vals):-1]
    z_grads = grad_vals[-1]
    for curr_W, curr_grad in zip(Ws_vals,Ws_grads):
        update(curr_W, curr_grad)
    for curr_bias, curr_grad in zip(biases_vals,biases_grads):
        update(curr_bias, curr_grad)
    #if np.mean(np.abs(z_grads)) > 1e-4:
    #    log(z_grads)
    update(zs,z_grads)

def partition(a):
    assert type(a) is np.ndarray
    assert a.shape[1] > minibatch_size, "a.shape[1] should be larger than the minibatch size. a.shape=%s"%str(a.shape)
    minibatches_num = int(a.shape[1] / minibatch_size)
    assert minibatches_num > 0
    off = lambda i : i * minibatch_size
    return [
        a[:,off(i):off(i+1)]
        for i
        in range(minibatches_num)
    ]

def partition_minibatches(Z,X):
    assert Z.shape[1] == X.shape[1], "Z and X have different lengths: %d and %d"%(Z.shape[1],X.shape[1])
    return list(zip(partition(Z),partition(X)))

def train(Z, X, Ws_vals, biases_vals, grad_fn,repeat=1):
    for zs,xs in tqdm(partition_minibatches(Z,X)*repeat,desc="training"):
        step(zs, xs, Ws_vals, biases_vals, grad_fn)

def nll_sum(Z, X, Ws_vals, biases_vals, nll_fn):
    ret = 0
    for zs,xs in tqdm(partition_minibatches(Z,X),desc="nll_sum"):
        curr, = nll_fn(*([zs, xs] + Ws_vals + biases_vals))
        ret += curr
    return ret

def reconstruction_error(Z, X, Ws_vals, biases_vals, inverse_fn, generate_fn):
    minibatches_means = []
    for _,xs in tqdm(partition_minibatches(Z,X)[:3],desc="reconstruction_error"):
        _zs_inverse, = inverse_fn(*([xs] + Ws_vals + biases_vals))
        log("_zs_inverse",_zs_inverse)
        curr_reconstructions, = generate_fn(*([_zs_inverse] + Ws_vals + biases_vals))
        differences = np.abs(xs - curr_reconstructions)
        minibatches_means.append(np.mean(differences))
    ret = np.mean(minibatches_means)
    return ret

def build_negative_log_likelihoods(zs,outputs,xs):
    error_term = 1/sigma_x * T.sum((xs-outputs)**2,axis=0)
    prior_term = 1/sigma_z * T.sum((zs)**2,axis=0)
    nlls = error_term + prior_term
    return nlls

def test_classifier(Z,Y):
    #classifier = sklearn.svm.SVC()
    log("training classifier..")
    classifier = sklearn.svm.SVC(
        kernel='rbf',
        max_iter=1000
    )
    # please notice the transpose '.T' operator: sklearn wants one datapoint per row
    classifier.fit(Z.T,Y[0,:])
    log("done. Scoring..")
    svc_score = classifier.score(Z.T,Y[0,:])
    log("SVC score: %s"%svc_score)

def generate_grid_samples(epoch,Ws_vals,biases_vals,generate_fn):
    log("generating samples from a grid")
    space1d = np.linspace(-2*sigma_z, 2*sigma_z, 20)
    mgs = np.meshgrid(*[space1d]*latent_dim)
    points = np.vstack([
        np.reshape(curr,-1)
        for curr
        in mgs
    ]).astype('float32')
    samples_l = []
    for curr in partition(points):
        samples_l.append( generate_fn(*([curr]+Ws_vals+biases_vals)) )
    samples = np.vstack(samples_l)
    filename = "grid_samples_epoch_%d.npy"%(epoch)
    np.save(filename, samples)
    log("done generating grid samples")

def generate_samples(epoch,Ws_vals,biases_vals,generate_fn,inverse_fn):
    log("generating a bunch of random samples")
    _zs_l = []
    for i in range(minibatch_size):
        _z = np.random.normal(np.array([0]*latent_dim),sigma_z).astype('float32')
        _zs_l.append(_z)
    _zs = np.vstack(_zs_l).T
    samples = generate_fn(*([_zs]+Ws_vals+biases_vals))
    log("generated samples. mean:",np.mean(samples),"std:",np.std(samples))
    #_zs_inverse = inverse_fn(*([samples[0]]+Ws_vals+biases_vals)) # FIXME: this 0 index
    log("_zs",_zs)
    #log("_zs_inverse!",_zs_inverse)
    filename = "random_samples_epoch_%d.npy"%(epoch)
    np.save(filename, samples)
    log("done generating random samples.")

def initial_weights_and_biases(x_dim):
    Ws_vals = []
    biases_vals = []
    dims1 = hidden_dims + [x_dim]
    dims2 = [ latent_dim ] + hidden_dims
    for curr in zip(dims1, dims2):
        xavier_var = 1./curr[0]
        W_vals_curr = (np.random.normal(0,xavier_var,curr).astype('float32'))
        biases_curr = (np.random.normal(0,xavier_var,(curr[0],1)).astype('float32'))
        Ws_vals.append(W_vals_curr)
        biases_vals.append(biases_curr)
    return Ws_vals, biases_vals

def main():
    global log
    global latent_dim
    global hidden_dims
    global minibatch_size
    global num_hidden_layers
    global activation_function
    assert len(sys.argv) > 1, "usage: %s harvest_dir"%(sys.argv[0])
    latent_dim = int(sys.argv[1])
    hidden_dims = list(map(int,sys.argv[2].split("_")))
    num_hidden_layers = len(hidden_dims)
    minibatch_size = int(sys.argv[3])
    activation_name = sys.argv[4]
    activation_function = possible_activations[activation_name]

    harvest_dir = "harvest_zdim{}_hdims_{}_minibatch_size_{}_activation_{}".format(
        latent_dim,
        sys.argv[2],
        minibatch_size,
        activation_name
    )
    np.set_printoptions(precision=4, suppress=True)
    X,Y,X_test,Y_test = load_data() # needs to be before cd
    try:
        os.mkdir(harvest_dir)
    except OSError as e: # directory already exists. It's ok.
        print(e)
    
    os.system("cp %s %s -vf"%(sys.argv[0],harvest_dir+"/"))
    os.chdir(harvest_dir)
    log = Logger()
    log("sys.argv",sys.argv)
    x_dim = X.shape[0]
    num_datapoints = X.shape[1]
    Z = (np.random.normal(0,sigma_z,(latent_dim,num_datapoints)).astype('float32'))
    Ws_vals, biases_vals = initial_weights_and_biases(x_dim)
    # set up
    zs, Ws, biases, outputs = build_net()
    xs = T.matrix('xs')
    #zs_inverted = build_net_inverse(xs,Ws,biases)
    nlls = build_negative_log_likelihoods(zs,outputs,xs)
    nll = T.sum(nlls,axis=0)

    for curr_W,curr_bias in zip(Ws,biases):
        weights_regularizer = 1/100 * T.sum((curr_W)**2) # FIXME: do proper derivation and variable naming
        bias_regularizer = 1/100 * T.sum(curr_bias**2)
        nll = nll + weights_regularizer + bias_regularizer
    grads = T.grad(nll,Ws+biases+[zs])
    #theano.pp(grad)

    def summary():
        total_nll = nll_sum(Z,X,Ws_vals,biases_vals,nll_fn)
        #_reconstruction_error = reconstruction_error(Z,X,Ws_vals,biases_vals,inverse_fn,generate_fn)
        log("epoch %d"%epoch)
        log("harvest_dir",harvest_dir)
        log("lr %f"%lr)
        log("total nll: {:,}".format(total_nll))
        #log("average reconstruction error: {:,}".format(_reconstruction_error))
        log("mean Z: {:,}".format(np.mean(Z)))
        log("mean abs Z: {:,}".format(np.mean(np.abs(Z))))
        log("std Z: {:,}".format(np.std(Z)))
        log("means Ws: %s"%([np.mean(curr) for curr in Ws_vals]))
        log("stds Ws: %s"%([np.std(curr) for curr in Ws_vals]))
        log("means biases: %s"%([np.mean(curr) for curr in biases_vals]))
        log("stds biases: %s"%([np.std(curr) for curr in biases_vals]))

    log("compiling theano grad_fn..")
    grad_fn = theano.function([zs, xs]+Ws+biases, grads)
    log("compiling theano nll_fn..")
    nll_fn = theano.function([zs, xs]+Ws+biases, [nll])
    log("compiling theano generate_fn..")
    generate_fn = theano.function([zs]+Ws+biases, [outputs])
    #log("compiling theano inverse_fn..")
    inverse_fn=None#inverse_fn = theano.function([xs]+Ws+biases, [zs_inverted])
    log("done. epochs loop..")

    def save():
        log("saving Z,Y,Ws,biases..")
        np.save("theano_decoder_Z.npy",Z)
        np.save("theano_decoder_Y.npy",Y)
        for i, (_w,_b) in enumerate(zip(Ws_vals,biases_vals)):
            np.save('theano_decoder_W_{}.npy'.format(i), _w)
            np.save('theano_decoder_bias_{}.npy'.format(i), _b)
        log("done saving.")

    # train
    for epoch in range(n_epochs):
        global lr
        lr = calculate_lr(epoch)
        Z,X,Y = shuffle(Z,X,Y)
        summary()
        if epoch % 5 == 0:
            generate_samples(epoch,Ws_vals,biases_vals,generate_fn,inverse_fn)
            generate_grid_samples(epoch,Ws_vals,biases_vals,generate_fn)
            test_classifier(Z,Y)
            save()
        train(Z,X,Ws_vals,biases_vals,grad_fn,repeat=repeat_training)
    log("epochs loop ended")
    summary()
if __name__=="__main__":
    main()

