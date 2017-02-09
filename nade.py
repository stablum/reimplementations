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
import lasagne
import math

sys.setrecursionlimit(20000)

optimizer = "debug"

if optimizer == "debug":
    theano_mode = 'DebugMode'
    theano.config.exception_verbosity="high"
    theano.config.optimizer='None'
    theano.config.floatX='float32'
elif optimizer == "gpu":
    theano.config.optimizer='fast_run'
    theano.config.openmp=False
    theano.config.openmp_elemwise_minsize=10
    assert theano.config.device=='gpu',theano.config.device
    theano.config.floatX='float32'

lr=0.0001#0.02
n_epochs = 10000
z_dim = None
x_dim = None
repeat_training=1
log = None
g = T.nnet.sigmoid
data_amplify = 1
data_offset = 0
minibatch_size = 64
regularizer_lambda=0.01

rs = np.random.RandomState(1234)
rng = theano.tensor.shared_randomstreams.RandomStreams(rs.randint(999999))
class Logger():
    def __init__(self,basename=""):
        self.filename = basename+"_"+str(time.time())+".log"
        self.f = open(self.filename,'w')

    def __call__(self, *args):
        print(*args, flush=True)
        print(*args,file=self.f, flush=True)

def make_nade(D,z_dim):
    log("make_nade with D={},z_dim={},g={}".format(D,z_dim,g))
    x = T.fmatrix('x')

    W_cols = []
    V_rows = []
    a_s = []
    hs = []
    bs = []
    c_vals = np.random.normal(0,1,size=(z_dim,1)).astype('float32')
    c = theano.shared(c_vals,name="c")
    a_s.append(c)
    p_x = 1
    for i in range(D):

        x_i = x[i,0]

        W_col_vals = np.random.normal(0,1,size=(z_dim,1)).astype('float32')
        W_col = theano.shared(W_col_vals,name="W_col_%d"%(i+1))
        W_cols.append(W_col)

        a_s.append(a_s[i] + W_col * x_i)

        hi = g(a_s[i+1])
        hs.append(hi)

        b_i_val = np.random.normal(0,1,size=(1,1)).astype('float32')
        b_i = theano.shared(b_i_val,name="b_i_%d"%(i+1))
        bs.append(b_i)

        V_row_vals = np.random.normal(0,1,size=(1,D)).astype('float32')
        V_row = theano.shared(V_row_vals,name="V_row_%d"%(i+1))

        p_x_cond = g(T.dot(V_row,hi) + b_i)
        p_x_cond_obs = x_i * p_x_cond + (1-x_i) * (1-p_x_cond)
        p_x = p_x * p_x_cond_obs

    return (W_cols,c,V_rows,bs),x,hs,p_x

def make_xcond(z_dummy,W):
    global g
    dot = T.dot(W,z_dummy)
    dot.name = "dot_generated"
    ret = g(dot)
    ret.name = "xcond"
    return ret

def make_xsample(xcond):
    global x_dim
    global rng
    ret = rng.binomial(n=1, p=xcond, size=(x_dim+1,1)) # +1 because bias/dummy
    return ret

def shuffle(X,Y):
    sel = np.arange(X.shape[1])
    np.random.shuffle(sel)
    X = X[:,sel]
    Y = Y[:,sel]
    return X,Y

def fix_data(features,labels):
    # please notice the transpose '.T' operator
    # in a neural network, the datapoints needs to be scattered across the columns
    # because dot product.
    X = (np.array(features).T.astype('float32')/255.)*data_amplify + data_offset
    Y = np.expand_dims(np.array(labels).astype('float32'),1).T
    return X,Y

def load_data():
    log("setting up mnist loader..")
    _mnist = mnist.MNIST(path='./python-mnist/data')
    log("loading training data..")
    X_train,Y_train = fix_data(*_mnist.load_training())
    log("X_train.shape=",X_train.shape,"Y_train.shape=",Y_train.shape)
    log("loading testing data..")
    X_test,Y_test = fix_data(*_mnist.load_testing())
    log("X_test.shape=",X_test.shape,"Y_test.shape=",Y_test.shape)
    return X_train[:,:], Y_train, X_test[:,:], Y_test

def step(xs, params, params_update_fn):
    nll = 0
    for i in range(xs.shape[1]):
        orig_x = xs[:,[i]]
        nll += params_update_fn(orig_x)
    return nll

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

def train(X, params, params_update_fn, repeat=1):
    nll = 0
    for xs in tqdm(partition(X)*repeat,desc="training"):
        # pseudo-contrastive err += step(xs, params, params_update_fn, zcond_fn, xcond_fn, params_contr_update_fn)
        nll += step(xs, params, params_update_fn)
    return nll

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

def draw_samples(epoch,xcond_fn):
    log("generating a bunch of random samples")
    samples = []
    for i in range(10):
        _z = np.random.normal(np.array([[0]*z_dim]),(i+1.)/2.).astype('float32')
        sample = xcond_fn(_z)
        samples.append(sample)
    samples_np = np.stack(samples,axis=2)
    filename = "random_samples_epoch_%d.npy"%(epoch)
    np.save(filename, samples_np)
    log("done generating random samples.")

def main():
    global log
    global z_dim
    global x_dim
    global minibatch_size
    global g
    assert len(sys.argv) > 1, "usage: %s z_dim"%(sys.argv[0])
    z_dim = int(sys.argv[1])

    random_int = np.random.randint(0,1000000)
    harvest_dir = "nade_harvest_zdim_{}_{}".format(
        z_dim,
        random_int
    )
    np.set_printoptions(precision=4, suppress=True)
    try:
        os.mkdir(harvest_dir)
    except OSError as e: # directory already exists. It's ok.
        print(e)

    log = Logger("{}/nadelog".format(harvest_dir)) # notice: before chdir to harvest_dir
    for curr in [sys.argv[0],"config.py","nade_job.sh","engage.sh"]:
        os.system("cp %s %s -vf"%(curr,harvest_dir+"/"))

    X,Y,X_test,Y_test = load_data()
    os.chdir(harvest_dir)
    log("sys.argv",sys.argv)
    x_dim = X.shape[0]
    num_datapoints = X.shape[1]
    # set up
    params,x,hs,p_x = make_nade(x_dim,z_dim)
    (W_cols,c,V_rows,bs) = params
    params_flat = W_cols+[c]+V_rows+bs

    nll = T.sum(- T.log(p_x))
    grads = []
    for param in tqdm(params_flat):
        print("gradient of param "+param.name)
        grad = T.grad(nll,param)
        grad.name = "grad_"#+param.name
        grads.append(grad)

    params_updates = lasagne.updates.adam(grads,params_flat,learning_rate=lr)
    # pseudo-contrastive params_update_fn = theano.function([x,z],[], updates=params_updates)
    params_update_fn = theano.function([x],nll, updates=params_updates)
    params_update_fn.name="params_update_fn"

    def generate_and_save(epoch):
        z_random = np.random.uniform(0,1,(z_dim,1)).astype('float32')
        xcond_gen = xcond_fn(z_random)
        filename = "nade_xcond_epoch_{0:04d}.npy".format(epoch)
        np.save(filename, xcond_gen)
        xsample_gen = xsample_fn(z_random)
        filename = "nade_xsample_epoch_{0:04d}.npy".format(epoch)
        np.save(filename, xsample_gen)

    def log_shared(qs):
        if type(qs) not in (list,tuple):
            qs = [q]
        for q in qs:
            log(q.name+": mean:{}, std:{}".format(
                np.mean(q.eval()),
                np.std(q.eval())
            ))

    def summary():
        log("epoch %d"%epoch)
        log("harvest_dir",harvest_dir)
        log("lr %f"%lr)
        log_shared(W_cols)

    log("done. epochs loop..")

    # train
    for epoch in range(n_epochs):
        X,Y = shuffle(X,Y)
        summary()
        #generate_and_save(epoch)
        nll = train(X,params,params_update_fn,repeat=repeat_training)

        log("nll",nll)
    log("epochs loop ended")
    summary()

if __name__=="__main__":
    main()

