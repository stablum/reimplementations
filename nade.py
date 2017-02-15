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

optimizer = "gpu"

if optimizer == "debug":
    theano_mode = 'DebugMode'
    theano.config.exception_verbosity="high"
    theano.config.optimizer='None'
    theano.config.floatX='float32'
elif optimizer == "gpu":
    theano.config.optimizer='fast_run'
    theano.config.openmp=True
    theano.config.openmp_elemwise_minsize=4
    #theano.config.device='gpu'
    theano.config.floatX='float32'
    theano.config.assert_no_cpu_op='raise'
    theano.config.allow_gc=False
    theano.config.nvcc.fastmath=True
    assert theano.config.device=='gpu',theano.config.device
elif optimizer == "fast_cpu":
    theano.config.optimizer='fast_run'
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

    c_vals = np.random.normal(0,1,size=(1,z_dim)).astype('float32')
    c = theano.shared(c_vals,name="c")
    p_x = 1

    def a_adder(W_col_T,x_i,acc):
        W_col_T.name = "W_col_T"
        prod = W_col_T * T.sum(x_i)
        prod.name = "prod"
        ret_T = acc.T + prod
        return ret_T.T

    """
    for i in range(D):
        W_col_vals = np.random.normal(0,1,size=(z_dim,1)).astype('float32')
        W_col = theano.shared(W_col_vals,name="W_col_%d"%(i+1))
        W_cols.append(W_col)
    """
    W_vals = np.random.normal(0,1,size=(z_dim,D)).astype('float32')
    W = theano.shared(W_vals,name="W")

    a_s_W,_u = theano.scan(
        fn=a_adder,
        outputs_info=c[0,:],
        sequences = [ W.T,
                      x
                    ]
        )
    a_s_excess = T.concatenate([c,a_s_W],axis=0)
    a_s = a_s_excess[:D,:]

    V_vals = np.random.normal(0,1,size=(D,z_dim)).astype('float32')
    V = theano.shared(V_vals,name="V")

    hs = g(a_s)

    b_val = np.random.normal(0,1,size=(D,1)).astype('float32')
    b = theano.shared(b_val,name="b")

    def scan_p_x_cond(V_row,hi,b_i):
        p_x_cond = g(T.dot(V_row,hi) + b_i)
        return p_x_cond

    p_x_cond,_u = theano.map(
        fn=scan_p_x_cond,
        sequences=[
            V,
            hs,
            b
        ]
    )

    def scan_p_x_cond_obs(x_i,p):
        ret = x_i * p + (1-x_i) * (1-p)
        return ret

    p_x_cond_obs,_u = theano.map(
        fn=scan_p_x_cond_obs,
        sequences=[
            x,
            p_x_cond
        ]
    )

    nll = - T.sum(T.log(p_x_cond_obs))

    p_x = T.prod(p_x_cond_obs)

    return (W,c,V,b),x,hs,p_x,nll,p_x_cond

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
        curr_nll,curr_p_x = params_update_fn(orig_x)
        nll += curr_nll
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
    _sum = 0
    for xs in tqdm(partition(X)*repeat,desc="training"):
        # pseudo-contrastive err += step(xs, params, params_update_fn, zcond_fn, xcond_fn, params_contr_update_fn)
        step_nll = step(xs, params, params_update_fn)
        average_nll = step_nll / xs.shape[1]
        log("step average nll:{}".format(average_nll))

        _sum += step_nll
    ret = _sum / X.shape[1]
    return ret

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

def test_nll(X_test,test_nll_fn,repeat=1):
    _sum = 0.
    for xs in tqdm(partition(X_test)*repeat,desc="testing"):
        for i in range(xs.shape[1]):
            x = xs[:,[i]]
            nll, p_x = test_nll_fn(x)
            _sum += nll

    ret = _sum/X_test.shape[1]
    return ret

def noise_nll(test_nll_fn):
    global x_dim
    _sum = 0.
    amount = 1000
    for i in tqdm(range(amount),desc="noise"):
        x = np.random.binomial(1,0.5,size=(x_dim,1)).astype('float32')
        nll, p_x = test_nll_fn(x)
        _sum += nll
    ret = _sum / amount
    return ret

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
    params,x,hs,p_x, nll, p_x_cond= make_nade(x_dim,z_dim)
    log("made nade")
    (W,c,V,b) = params

    grads = []
    for param in tqdm(params):
        log("gradient of param "+param.name)
        grad = T.grad(nll,param)
        grad.name = "grad_"#+param.name
        grads.append(grad)

    params_updates = lasagne.updates.adam(grads,params,learning_rate=lr)
    # pseudo-contrastive params_update_fn = theano.function([x,z],[], updates=params_updates)
    params_update_fn = theano.function([x],[nll,p_x], updates=params_updates)
    params_update_fn.name="params_update_fn"

    test_nll_fn = theano.function([x],[nll,p_x])
    gen_fn = theano.function([hs],p_x_cond)
    def generate_and_save(epoch):
        hs_random = np.random.uniform(0,1,(x_dim,z_dim)).astype('float32')
        samples = gen_fn(hs_random)
        filename = "nade_samples_epoch_{0:04d}.npy".format(epoch)
        np.save(filename,samples)

    def log_shared(qs):
        if type(qs) not in (list,tuple):
            qs = [qs]
        for q in qs:
            log(q.name+": mean:{}, std:{}".format(
                np.mean(q.eval()),
                np.std(q.eval())
            ))

    def summary():
        log("epoch %d"%epoch)
        log("harvest_dir",harvest_dir)
        log("lr %f"%lr)
        log_shared(W)

    log("done. epochs loop..")

    # train
    for epoch in range(n_epochs):
        X,Y = shuffle(X,Y)
        summary()
        generate_and_save(epoch)
        nll_noise = noise_nll(test_nll_fn)
        log("epoch average noise nll:", nll_noise)
        nll_test = test_nll(X_test,test_nll_fn)
        log("epoch average test nll:", nll_test)
        nll = train(X,params,params_update_fn,repeat=repeat_training)
        log("epoch average training nll:", nll)
    log("epochs loop ended")
    summary()

if __name__=="__main__":
    main()

