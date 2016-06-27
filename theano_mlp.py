#!/usr/bin/env python3

import theano
from theano import tensor as T
import pandas as pd
from tqdm import trange,tqdm
import numpy as np
from sklearn.preprocessing import normalize
#theano.config.exception_verbosity="high"
theano.config.optimizer='fast_run'
train_filename = 'adult/adult.data'
test_filename = 'adult/adult.test'
lr = 0.0001
n_epochs = 100

def load_file(filename):
    skiprows = 1 if 'test' in filename else 0
    df = pd.read_csv(filename,skiprows=skiprows,header=None)
    df = df.fillna("not_available")
    return df

def shuffle(X,Y):
    sel = np.arange(X.shape[0])
    np.random.shuffle(sel)
    X = X[sel,:]
    Y = Y[sel,:]
    return X,Y

def load_data():
    df_train = load_file(train_filename)
    df_test = load_file(test_filename)
    df = pd.concat([df_train,df_test])
    ddf = pd.get_dummies(df)
    XY = ddf.as_matrix()
    X = XY[:,:-2]
    Y = XY[:,[-2]]
    X = normalize(X,axis=0)
    X,Y = shuffle(X,Y)
    cutpoint = int(X.shape[0]*0.75)
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
    out1 = T.nnet.sigmoid(lin1)
    W2 = T.dmatrix('W2')
    lin2 = T.dot(W2, out1)
    lin2.name = 'lin2'
    out2 = T.nnet.sigmoid(lin2)
    W3 = T.dmatrix('W3')
    lin3 = T.dot(W3, out2)
    lin3.name = 'lin3'
    out3 = T.nnet.sigmoid(lin3)
    W4 = T.dmatrix('W4')
    lin4 = T.dot(W4, out3)
    lin4.name = 'lin4'
    out4 = T.nnet.sigmoid(lin4)

    return inputs, [W1,W2,W3,W4], out4

def build_cost(outputs):
    target = T.dvector('target')
    diff = outputs - target
    pows = T.pow(diff,2)
    _sum = T.sum(pows)
    cost = T.sqrt(_sum)
    return cost,target

def step(x, y, Ws_vals, grad_fn):
    grad_vals = grad_fn(*([x, y] + Ws_vals))
    #print("grad_vals mean:%s W_vals mean:%s linear:%s output_vals:%s"%(
    #    np.mean(grad_vals),
    #    np.mean(W_vals),
    #    np.mean(linear),
    #    np.mean(output_vals)
    #))
    ret = []
    for curr_W, curr_grad in zip(Ws_vals,grad_vals):
        curr_W -= lr * curr_grad
        ret.append(curr_W)
    return ret

def train(X, Y, Ws_vals, grad_fn):
    for x,y in tqdm(list(zip(X,Y))):
        Ws_vals = step(x, y, Ws_vals, grad_fn)
    return Ws_vals

def error(X, Y, Ws_vals, cost_fn):
    total = 0.
    for x,y in tqdm(list(zip(X,Y))):
        total += cost_fn(*([x,y]+Ws_vals))[0]
    ret = total / X.shape[0]
    return ret

def main():
    np.set_printoptions(precision=4, suppress=True)
    X,Y,X_test,Y_test = load_data()
    n_features = X.shape[1]
    target_dim = Y.shape[1]
    H1=31 # dimensionality of hidden layer
    H2=17
    H3=11
    W1_vals = np.random.random((H1,n_features))*0.1
    W2_vals = np.random.random((H2,H1))*0.1
    W3_vals = np.random.random((H3,H2))*0.1
    W4_vals = np.random.random((target_dim,H3))*0.1
    Ws_vals = [W1_vals,W2_vals,W3_vals,W4_vals]
    # set up
    inputs, Ws, outputs = build_net()
    cost, target = build_cost(outputs)
    grads = T.grad(cost,Ws)
    #theano.pp(grad)

    # FIXME: remove 'linear' from the output variables
    grad_fn = theano.function([inputs, target]+Ws, grads)
    cost_fn = theano.function([inputs, target]+Ws, [cost])

    # train
    for epoch in range(n_epochs):
        X,Y = shuffle(X,Y)
        Ws_vals = train(X,Y,Ws_vals,grad_fn)
        train_error = error(X,Y,Ws_vals,cost_fn)
        test_error = error(X_test,Y_test,Ws_vals,cost_fn)
        print("epoch %d"%epoch)
        print("train error: %f"%train_error)
        print("test error: %f\n"%test_error)

if __name__=="__main__":
    main()

