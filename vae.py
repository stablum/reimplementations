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

optimizer = "gpu"

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

lr=0.02
n_epochs = 10000
data_amplify = 0.5
data_offset = 0.25
x_sigma = 1
z_dim = None
x_dim = None
hid_dims = None
activation_function = None
minibatch_size = None
repeat_training=1
g=None # activation function

possible_activations = {
    'sigmoid': T.nnet.sigmoid,

    # 2.37 seems to make a sigmoid a good approximation for erf(x),
    'pseudogelu': lambda x: x * T.nnet.sigmoid(x*2.37),

    'gelu': lambda x : x*T.erf(x),
    'elu': T.nnet.elu,
    'relu': T.nnet.relu,
    'linear': lasagne.nonlinearities.linear
}

class Logger():
    def __init__(self,basename=""):
        self.filename = basename+"_"+str(time.time())+".log"
        self.f = open(self.filename,'w')

    def __call__(self, *args):
        print(*args, flush=True)
        print(*args,file=self.f, flush=True)

log = None

def make_net(input_var,in_dim,hid_dim,out_dim,name="",output_nonlinearity=g):
    input_var_reshaped = input_var.reshape((1, in_dim))
    l_in = lasagne.layers.InputLayer((1,in_dim),input_var=input_var_reshaped,name=name+"_in")
    l_hid = lasagne.layers.DenseLayer(l_in,hid_dim,nonlinearity=g,name=name+"_hid")
    l_out = lasagne.layers.DenseLayer(l_hid,out_dim,nonlinearity=output_nonlinearity,name=name+"_out")
    net_output = lasagne.layers.get_output(l_out)
    net_params = lasagne.layers.get_all_params([l_in,l_hid,l_out])
    return net_output, net_params

def make_vae(x_dim,z_dim,hid_dim):
    print("make_vae with x_dim={},z_dim={},hid_dim={},g={}".format(x_dim,z_dim,hid_dim,g))
    x_orig = T.fmatrix('x_orig')
    z_dist,recog_params = make_net(
        x_orig,
        x_dim,
        hid_dim,
        z_dim*2,
        name="recog",
        output_nonlinearity=lasagne.nonlinearities.linear
    )
    z_dist.name="z_dist"
    epsilon = T.fvector('epsilon')
    epsilon.name = 'epsilon'
    z_mu = z_dist[:,0:z_dim]
    z_mu.name = 'z_mu'

    log_z_sigma = z_dist[:,z_dim:z_dim*2]
    log_z_sigma.name = "log_z_sigma"
    z_sigma = T.exp(log_z_sigma)

    #z_sigma = z_dist[:,z_dim:z_dim*2]
    z_sigma.name = 'z_sigma'
    z_sample = z_mu + (epsilon * z_sigma)
    z_sample.name = 'z_sample'
    z_sample_reshaped = z_sample.reshape((z_dim,))
    x_out,gener_params = make_net(z_sample_reshaped,z_dim,hid_dim,x_dim,name="gener")
    params = recog_params + gener_params
    return params,x_orig,x_out,z_mu,z_sigma,z_sample,z_dist,epsilon

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
    print("setting up mnist loader..")
    _mnist = mnist.MNIST(path='./python-mnist/data')
    print("loading training data..")
    X_train,Y_train = fix_data(*_mnist.load_training())
    print("X_train.shape=",X_train.shape,"Y_train.shape=",Y_train.shape)
    print("loading testing data..")
    X_test,Y_test = fix_data(*_mnist.load_testing())
    print("X_test.shape=",X_test.shape,"Y_test.shape=",Y_test.shape)
    return X_train, Y_train, X_test, Y_test

def update(learnable, grad):
    learnable -= lr * grad

def sample_epsilon():
    return np.random.normal(0,1,(z_dim,)).astype('float32')

def step(xs, params, params_update_fn):
    for i in range(xs.shape[1]):
        x = xs[:,[i]].T
        epsilon = sample_epsilon()
        params_update_fn(x,epsilon)

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
    for xs in tqdm(partition(X)*repeat,desc="training"):
        step(xs, params, params_update_fn)

def obj_sum(X,obj_fn):
    ret = 0
    obj_min = float('inf')
    obj_max = float('-inf')
    objs = []
    z_sigmas = []
    z_mus = []
    z_samples = []
    for i in tqdm(range(X.shape[1]),desc="obj_sum"):
        x = X[:,[i]].T
        epsilon = sample_epsilon()
        obj_quantities = obj_fn(x,epsilon)
        obj = obj_quantities[0]
        z_sigmas.append(obj_quantities[10])
        z_mus.append(obj_quantities[9])
        z_samples.append(obj_quantities[8])
        if math.isinf(obj) or i==0:
            for i,q in enumerate(obj_quantities):
                print(i,':',q)
        objs.append(obj)
        ret += obj
        obj_min = min(obj_min,obj)
        obj_max = max(obj_max,obj)
    obj_median = np.median(objs)
    z_sigmas_mean = np.mean(z_sigmas,axis=0)
    z_sigmas_std = np.std(z_sigmas,axis=0)
    z_mus_mean = np.mean(z_mus,axis=0)
    z_mus_std = np.std(z_mus,axis=0)
    z_samples_mean = np.mean(z_samples,axis=0)
    z_samples_std = np.std(z_samples,axis=0)
    return ret,obj_min,obj_max,obj_median,z_sigmas_mean,z_sigmas_std,z_mus_mean,z_mus_std,z_samples_mean,z_samples_std

def kl_normal_diagonal(mu1,sigma_diag1,mu2,sigma_diag2,dim):
    det1 = T.prod(sigma_diag1)
    det2 = T.prod(sigma_diag2)
    inv_sigma_diag2 = 1/sigma_diag_2
    mu_diff = mu2-mu1
    ret = 0.5 * (
        log(det2/det1)
        - dim
        + T.sum(inv_sigma_diag2*sigma_diag1)
        + T.dot(T.dot(mu_diff.T,inv_sigma_diag2),mu_diff)
    )
    return ret

def kl_normal_diagonal_vs_unit(mu1,sigma_diag1,dim):
    # KL divergence of a multivariate normal over a normal with 0 mean and I cov
    log_det1 = T.sum(T.log(sigma_diag1)) #sum log is better than log prod
    mu_diff = -mu1
    ret = 0.5 * (
        - log_det1
        - dim
        + T.sum(sigma_diag1) # trace
        + T.sum(mu_diff**2) # mu^T mu
    )
    return ret

def build_obj(z_sample,z_mu,z_sigma,x_orig,x_out):
    z_sigma_fixed = z_sigma
    z_sigma_inv = 1/(z_sigma_fixed)
    det_z_sigma = T.prod(z_sigma)
    C = 1./(T.sqrt(((2*np.pi)**z_dim) * det_z_sigma))
    log_q_z_given_x = - 0.5*T.dot(z_sigma_inv, ((z_sample-z_mu)**2).T) + T.log(C) # log(C) can be omitted
    q_z_given_x = C * T.exp(log_q_z_given_x)
    log_p_x_given_z = -(1/(x_sigma))*(((x_orig-x_out)**2).sum()) # because p(x|z) is gaussian
    log_p_z = - (z_sample**2).sum() # gaussian prior with mean 0 and cov I
    reconstruction_error_const = (0.5*(x_dim*np.log(np.pi)+1)).astype('float32')
    reconstruction_error_proper = 0.5*T.sum((x_orig-x_out)**2)
    reconstruction_error = reconstruction_error_const + reconstruction_error_proper
    regularizer = kl_normal_diagonal_vs_unit(z_mu,z_sigma,z_dim)
    obj = reconstruction_error + regularizer
    obj_scalar = obj.reshape((),ndim=0)
    return obj_scalar,[
        reconstruction_error, #1
        regularizer,#2
        log_q_z_given_x,#3
        det_z_sigma,#4
        q_z_given_x,#5
        log_p_x_given_z,#6
        log_p_z,#7
        z_sample,#8
        z_mu,#9
        z_sigma,#10,
        z_sigma_inv,#11
        z_sigma_fixed,#12
        C,#13
        reconstruction_error_proper,#14
    ]

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

def generate_samples(epoch,generate_fn):
    log("generating a bunch of random samples")
    samples = []
    for i in range(10):
        _z = np.random.normal(np.array([[0]*z_dim]),(i+1.)/2.).astype('float32')
        sample = generate_fn(_z)
        samples.append(sample)
    samples_np = np.stack(samples,axis=2)
    filename = "random_samples_epoch_%d.npy"%(epoch)
    np.save(filename, samples_np)
    log("done generating random samples.")

def main():
    global log
    global z_dim
    global x_dim
    global hid_dims
    global minibatch_size
    global activation_function
    global g
    assert len(sys.argv) > 1, "usage: %s harvest_dir"%(sys.argv[0])
    z_dim = int(sys.argv[1])
    hid_dim = int(sys.argv[2])
    minibatch_size = int(sys.argv[3])
    activation_name = sys.argv[4]
    g = possible_activations[activation_name]

    harvest_dir = "harvest_zdim{}_hdim_{}_minibatch_size_{}_activation_{}".format(
        z_dim,
        hid_dim,
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
    # set up
    params,x_orig,x_out,z_mu,z_sigma,z_sample,z_dist,epsilon = make_vae(x_dim,z_dim,hid_dim)
    obj,other_quantities = build_obj(z_sample,z_mu,z_sigma,x_orig,x_out)

    obj_fn = theano.function([x_orig,epsilon],[obj]+other_quantities+[z_dist])
    #minibatch_obj = T.sum(objs,axis=0)

    grads_params = [
        T.grad(obj,curr)
        for curr
        in params
    ]
    params_updates = lasagne.updates.adam(grads_params,params,learning_rate=lr)
    params_update_fn = theano.function([x_orig,epsilon],[], updates=params_updates)

    generate_fn = theano.function([z_sample],[x_out])

    def summary():
        total_obj,obj_min,obj_max,obj_median,z_sigmas_mean,z_sigmas_std,z_mus_mean,z_mus_std,z_samples_mean,z_samples_std = obj_sum(X,obj_fn)
        print(z_sigmas_mean)
        log("epoch %d"%epoch)
        log("harvest_dir",harvest_dir)
        log("lr %f"%lr)
        log("total_obj: {}".format(total_obj))
        log("obj_min: {}".format(obj_min))
        log("obj_max: {}".format(obj_max))
        log("obj_median: {}".format(obj_median))
        log("z_sigmas_mean",z_sigmas_mean)
        log("z_sigmas_std",z_sigmas_std)
        log("z_mus_mean",z_mus_mean)
        log("z_mus_std",z_mus_std)
        log("z_samples_mean",z_samples_mean)
        log("z_samples_std",z_samples_std)
        #log("total nll: {:,}".format(total_nll))

    log("done. epochs loop..")

    def save():
        log("saving Y,Ws,biases..")
        np.save("theano_decoder_Z.npy",Z)
        np.save("theano_decoder_Y.npy",Y)
        for i, (_w,_b) in enumerate(zip(Ws_vals,biases_vals)):
            np.save('theano_decoder_W_{}.npy'.format(i), _w)
            np.save('theano_decoder_bias_{}.npy'.format(i), _b)
        log("done saving.")

    # train
    for epoch in range(n_epochs):
        X,Y = shuffle(X,Y)
        summary()
        if epoch % 1 == 0:
            generate_samples(epoch,generate_fn)
            #save()
        train(X,params,params_update_fn,repeat=repeat_training)
    log("epochs loop ended")
    summary()
if __name__=="__main__":
    main()

