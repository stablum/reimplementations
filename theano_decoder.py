#!/usr/bin/env python3

import theano
from theano import tensor as T
import pandas as pd
from tqdm import trange,tqdm
import numpy as np
from sklearn.preprocessing import normalize
import sklearn.svm

theano.config.exception_verbosity="high"
theano.config.optimizer='None'
train_filename = 'ionosphere/ionosphere.data'
lr_begin = 0.5
lr_annealing_T=1
lr=None
n_epochs = 10000
sigma_x = 1.
sigma_z = 1e4
nonlinearity = T.nnet.sigmoid

def calculate_lr(t):
    # decaying learning rate with annealing
    # see: https://www.willamette.edu/~gorr/classes/cs449/momrate.html
    ret = lr_begin / (
        1. + float(t)/lr_annealing_T
    )
    return ret

def load_file(filename):
    skiprows = 1 if 'test' in filename else 0
    df = pd.read_csv(filename,skiprows=skiprows,header=None)
    df = df.fillna("not_available")
    return df

def shuffle(Z,X,Y):
    sel = np.arange(X.shape[0])
    np.random.shuffle(sel)
    X = X[sel,:]
    Z = Z[sel,:]
    Y = Y[sel,:]
    return Z,X,Y

def load_data():
    df = load_file(train_filename)
    #ddf = pd.get_dummies(df)
    XY = df.as_matrix()
    X = XY[:,:-1].astype('float')
    Y = np.expand_dims((df.iloc[:,-1] == 'g').astype('float').as_matrix(),1)
    #Y = XY[:,[-1]]
    #X = normalize(X,axis=0)
    X,Y,_ = shuffle(X,Y,Y)
    cutpoint = int(X.shape[0]*0.95)
    X_train = X[:cutpoint,:]
    Y_train = Y[:cutpoint,:]
    X_test = X[cutpoint:,:]
    Y_test = Y[cutpoint:,:]
    return X_train, Y_train, X_test, Y_test

def build_net():
    inputs = T.dvector('inputs')
    W1 = T.dmatrix('W1')
    lin1 = T.dot(W1, inputs)
    lin1.name = 'lin1'
    out1 = nonlinearity(lin1)
    W2 = T.dmatrix('W2')
    lin2 = T.dot(W2, out1)
    lin2.name = 'lin2'
    out2 = nonlinearity(lin2)
    W3 = T.dmatrix('W3')
    lin3 = T.dot(W3, out2)
    lin3.name = 'lin3'
    out3 = nonlinearity(lin3)

    return inputs, [W1,W2,W3], out3

def update(learnable, grad):
    learnable -= lr * grad

def step(z, x, Ws_vals, grad_fn):
    grad_vals = grad_fn(*([z, x] + Ws_vals))
    Ws_grads = grad_vals[:-1]
    z_grads = grad_vals[-1]
    for curr_W, curr_grad in zip(Ws_vals,Ws_grads):
        update(curr_W, curr_grad)
    #if np.mean(np.abs(z_grads)) > 1e-4:
    #    print(z_grads)
    update(z,z_grads)

def train(Z, X, Ws_vals, grad_fn,repeat=1):
    for z,x in tqdm(list(zip(Z,X))*repeat):
        step(z, x, Ws_vals, grad_fn)

def nll_sum(Z, X, Ws_vals, nll_fn):
    ret = 0
    for z,x in tqdm(list(zip(Z,X))):
        curr, = nll_fn(*([z, x] + Ws_vals))
        ret += curr
    return ret

def build_negative_log_likelihood(z,outputs,x):
    error_term = 1/sigma_x * T.sum((x-outputs)**2)
    prior_term = 1/sigma_z * T.sum((z)**2)
    nll = error_term + prior_term
    return nll

def test_classifier(Z,Y):
    #classifier = sklearn.svm.SVC()
    classifier = sklearn.svm.SVC(kernel='linear')
    classifier.fit(Z,Y[:,0])
    svc_score = classifier.score(Z,Y[:,0])
    print("SVC score: %s"%svc_score)

def main():
    np.set_printoptions(precision=4, suppress=True)
    X,Y,X_test,Y_test = load_data()
    x_dim = X.shape[1]
    latent_dim=2
    Z = (np.random.random((X.shape[0],latent_dim))-0.5) * 1
    H1=5# dimensionality of hidden layer
    H2=17# dimensionality of hidden layer
    W1_vals = np.random.random((H1,latent_dim))*0.1
    W2_vals = np.random.random((H2,H1))*0.1
    W3_vals = np.random.random((x_dim,H2))*0.1
    Ws_vals = [W1_vals,W2_vals,W3_vals]
    # set up
    z, Ws, outputs = build_net()
    x = T.dvector('x')
    nll = build_negative_log_likelihood(z,outputs,x)
    grads = T.grad(nll,Ws+[z])
    #theano.pp(grad)

    grad_fn = theano.function([z, x]+Ws, grads)
    nll_fn = theano.function([z, x]+Ws, [nll])

    def summary():
        total_nll = nll_sum(Z,X,Ws_vals,nll_fn)
        print("epoch %d"%epoch)
        print("lr %f"%lr)
        print("total nll: %f"%total_nll)
        print("mean Z: %f"%np.mean(Z))
        print("mean abs Z: %f"%np.mean(np.abs(Z)))
        print("std Z: %f"%np.std(Z))
        print("means Ws: %s"%([np.mean(curr) for curr in Ws_vals]))
        print("stds Ws: %s"%([np.std(curr) for curr in Ws_vals]))

    # train
    for epoch in range(n_epochs):
        global lr
        lr = calculate_lr(epoch)
        Z,X,Y = shuffle(Z,X,Y)
        summary()
        test_classifier(Z,Y)
        train(Z,X,Ws_vals,grad_fn,repeat=1)
        np.save("theano_decoder_Z.npy",Z)
    summary()
if __name__=="__main__":
    main()

