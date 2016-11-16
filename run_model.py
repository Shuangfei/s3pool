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

from build_net import build_net
from load_data import load_data, get_iterator


def main(dataset='cifar10', option='standard', arch='resnet-3', flip=0, crop=0,
         lr=1, num_epochs=200, num_for_train=0, lowerlr_at=-1, save_model=False, load_model='none', num_ensembles=0):
    
    print "Loading data..."
    data = load_data(dataset)
    X_train = data['X_train']
    Y_train = data['Y_train']
    X_test = data['X_test']
    Y_test = data['Y_test']
    ny = {'cifar10':10, 'cifar100':100, 'stl10':10}
    iterate_minibatches = get_iterator(dataset)
    
    input_var, target_var, network = build_net(option=option, arch=arch, ny=ny[dataset])
    print "number of parameters in model: %d" % lasagne.layers.count_params(network, trainable=True)
    sys.stdout.flush()

    def get_loss_and_acc(prediction, target_var):
        loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
        loss = loss.mean()
        acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                     dtype=theano.config.floatX)
        return loss, acc

    if num_epochs:
        prediction = lasagne.layers.get_output(network, deterministic=False)
        loss, acc = get_loss_and_acc(prediction, target_var)
        all_layers = lasagne.layers.get_all_layers(network)
        params = lasagne.layers.get_all_params(network, trainable=True)
        sh_lr = theano.shared(lasagne.utils.floatX(lr))
        updates = lasagne.updates.adadelta(loss, params, learning_rate=sh_lr)
        
        train_fn = theano.function([input_var, target_var], [loss, acc], updates=updates)


    if num_ensembles == 0:
        test_prediction = lasagne.layers.get_output(network, deterministic=True)
    else:
        test_prediction = []
        for i in range(num_ensembles):
            test_prediction.append(lasagne.layers.get_output(network, deterministic=False,
                                                             batch_norm_use_averages=True,
                                                             batch_norm_update_averages=False))
        test_prediction = sum(test_prediction) / num_ensembles

    val_fn = theano.function([input_var, target_var],
                             list(get_loss_and_acc(test_prediction, target_var)))

    if load_model is None or load_model == 'none':
        epochs_seen = 0
    else:
        print 'loading from', load_model
        sys.stdout.flush()
        with np.load(load_model) as f:
             param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(network, param_values)
        epochs_seen = int(load_model.strip('.npz').split('_')[-1].lstrip('epochs'))

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
            train_acc = 0
            train_batches = 0
            start_time = time.time()
            for batch in iterate_minibatches(X_train, Y_train, 128, flip=flip, crop=crop,
                                             num=num_for_train):
                inputs, targets = batch
                err, acc = train_fn(inputs, targets)
                train_err += err
                train_acc += acc
                train_batches += 1

            # And a full pass over the validation data:
            val_err = 0
            val_acc = 0
            val_batches = 0
            for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1

            # Then we print the results for this epoch:
            print "Epoch {} of {} took {:.3f}s".format(
                epoch + 1, num_epochs, time.time() - start_time)
            print "  training loss:\t\t{:.6f}".format(train_err / train_batches)
            print "  training  accuracy:\t\t{:.2f} %".format(
                train_acc / train_batches * 100)

            print "  validation loss:\t\t{:.6f}".format(val_err / val_batches)
            print "  validation accuracy:\t\t{:.2f} %".format(
                val_acc / val_batches * 100)
            sys.stdout.flush()
            
        if save_model:
            name = 'models/%s_%s_%s_flip%d_crop%d_epochs%s.npz' \
                   % (dataset, arch, option, flip, crop, epochs_seen + num_epochs)
            print 'saving model to %s' % (name)
            np.savez(name, *lasagne.layers.get_all_param_values(network))

    # Calculate validation error of model:
    test_err = 0
    test_acc = 0
    test_batches = 0
    for batch in iterate_minibatches(X_test, Y_test, 500, shuffle=False):
        inputs, targets = batch
        err, acc = val_fn(inputs, targets)
        test_err += err
        test_acc += acc
        test_batches += 1
    print "Final results:"
    print "  test loss:\t\t\t{:.6f}".format(test_err / test_batches)
    print "  test accuracy:\t\t{:.2f} %".format(
        test_acc / test_batches * 100)
    sys.stdout.flush()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-dataset', type=str, default='cifar10')
    parser.add_argument('-option', type=str, default='standard')
    parser.add_argument('-arch', type=str, default='resnet-5')
    parser.add_argument('-flip', type=int, default=0)
    parser.add_argument('-crop', type=int, default=0)
    parser.add_argument('-lr', type=float, default=1.)
    parser.add_argument('-num_epochs', type=int, default=200)
    parser.add_argument('-num_for_train', type=int, default=0)
    parser.add_argument('-lowerlr_at', type=int, default=-1)
    parser.add_argument('-save_model', type=int, default=1)
    parser.add_argument('-load_model', type=str, default='none')
    parser.add_argument('-num_ensembles', type=int, default=0)
    config = vars(parser.parse_args())
    print config
    main(**config)
