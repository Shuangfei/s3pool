#!/usr/bin/env python
import sys
import os
import time
import string
import random
import pickle

import numpy as np
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *
from lib.vis import color_grid_vis

from build_net import build_net
from load_data import load_data, get_iterator


def main(dataset='cifar10', option='stochatic-16-8', arch='nin', vis_layer=0,
         lr=1, num_epochs=200, lowerlr_at=-1, load_model='none', **kwargs):
    
    print "Loading data..."
    data = load_data(dataset)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    ny = {'cifar10':10, 'cifar100':100, 'svhn':10, 'food101':101}
    iterate_minibatches = get_iterator(dataset)
    
    input_var, target_var, network = build_net(option=option, arch=arch, ny=ny[dataset], visualize=True)

    pred_net = network[0]
    inv_net = network[vis_layer+1]

    if num_epochs:
        params = lasagne.layers.get_all_params(pred_net, trainable=True)
        print 'loading from', load_model
        sys.stdout.flush()
        with np.load(load_model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(pred_net, param_values)
        
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        recon = get_output(inv_net, deterministic=True,
                           batch_norm_use_averages=False,
                           batch_norm_update_averages=True
        )
        inv_params = get_all_params(inv_net, trainable=True)
        inv_params = [p for p in inv_params if p not in params]
        loss = lasagne.objectives.squared_error(input_var, recon).mean(axis=0).sum()
        updates = lasagne.updates.adadelta(loss, inv_params, learning_rate=sh_lr)
        train_fn = theano.function([input_var], loss, updates=updates)
        recon_static = get_output(inv_net, deterministic=True)
        recon_stochastic = get_output(inv_net, deterministic=False,
                                      batch_norm_use_averages=True,
                                      batch_norm_update_averages=False
        )
        static_recon_fn = theano.function([input_var], recon_static)
        stochastic_recon_fn = theano.function([input_var], recon_stochastic)

    if num_epochs:
        # launch the training loop
        print "Starting training..."
        sys.stdout.flush()
        # We iterate over epochs:
        for epoch in range(num_epochs):
            if epoch == lowerlr_at:
                new_lr = sh_lr.get_value() * 0.1
                print "reducing lr to "+str(new_lr)
                sh_lr.set_value(lasagne.utils.floatX(new_lr))

            train_err = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, 128, flip=False, crop=False):
                inputs, targets = batch
                err = train_fn(inputs)
                train_err += err
                train_batches += 1

            # Then we print the results for this epoch:
            print "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time)
            print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
            sys.stdout.flush()

    sample = []
    for i in range(ny[dataset]):
        idx = np.where(Y_test[:1000] == i)[0][:5]
        sample.append(X_test[idx])
    sample = np.concatenate(sample, axis=0)
    color_grid_vis(sample.transpose(0, 2, 3, 1), (10, 5), arch+'_sample%d.png' % (vis_layer))
    color_grid_vis(static_recon_fn(sample).transpose(0, 2, 3, 1), (10, 5), arch+'_static_recon%d.png' % (vis_layer))
    color_grid_vis(stochastic_recon_fn(sample).transpose(0, 2, 3, 1), (10, 5), arch+'_stochastic_recon%d.png' % (vis_layer))
    
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-option', type=str, default='standard')
    parser.add_argument('-arch', type=str, default='resnet-5')
    parser.add_argument('-vis_layer', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1.)
    parser.add_argument('-num_epochs', type=int, default=200)
    parser.add_argument('-lowerlr_at', type=int, default=-1)
    parser.add_argument('-load_model', type=str, default='none')
    config = vars(parser.parse_args())
    print config
    main(**config)
