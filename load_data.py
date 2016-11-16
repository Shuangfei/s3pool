import sys
import os
import time
import string
import random
import cPickle as pkl

import numpy as np
from scipy.io import loadmat
import theano
import theano.tensor as T
import lasagne
from lasagne.layers import *

def unpickle(file):
    fo = open(file, 'rb')
    dict = pkl.load(fo)
    fo.close()
    return dict

def load_cifar10(datapath='datasets/cifar-10-batches-py/'):
    xs = []
    ys = []
    for j in range(5):
      d = unpickle(datapath+'data_batch_%s'%(j+1))
      x = d['data']
      y = d['labels']
      xs.append(x)
      ys.append(y)

    d = unpickle(datapath+'test_batch')
    xs.append(d['data'])
    ys.append(d['labels'])

    x = np.concatenate(xs)/np.float32(255)
    y = np.concatenate(ys)
    x = np.dstack((x[:, :1024], x[:, 1024:2048], x[:, 2048:]))
    x = x.reshape((x.shape[0], 32, 32, 3)).transpose(0,3,1,2)

    pixel_mean = np.mean(x[0:50000],axis=0)
    x -= pixel_mean

    X_train = x[0:50000,:,:,:]
    Y_train = y[0:50000]

    X_test = x[50000:,:,:,:]
    Y_test = y[50000:]

    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

def load_cifar100(data_dir='datasets/cifar-100-python/'):
    train_file = 'train'
    test_file = 'test'
    train_data = unpickle(data_dir + train_file)
    test_data = unpickle(data_dir + test_file)

    X_train = train_data['data'].reshape(-1, 3, 32, 32) / 255.
    Y_train = np.array(train_data['fine_labels'])
    
    X_test = test_data['data'].reshape(-1, 3, 32, 32) / 255.
    Y_test = np.array(test_data['fine_labels'])

    pixel_mean = np.mean(X_train,axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    
    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)

def load_stl10(data_dir='datasets/stl10_binary/'):
    def read_all_images(path_to_data):
        with open(path_to_data, 'rb') as f:
            everything = np.fromfile(f, dtype=np.uint8)
            images = np.reshape(everything, (-1, 3, 96, 96)) / 255.
        return images

    def read_labels(path_to_labels):
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
        labels[np.where(labels == 10)[0]] = 0
        return labels

    X_train = read_all_images(data_dir + 'train_X.bin')
    Y_train = read_labels(data_dir + 'train_y.bin')
    X_test = read_all_images(data_dir + 'test_X.bin')
    Y_test = read_labels(data_dir + 'test_y.bin')
    pixel_mean = np.mean(X_train,axis=0)
    X_train -= pixel_mean
    X_test -= pixel_mean
    
    return dict(
        X_train=lasagne.utils.floatX(X_train),
        Y_train=Y_train.astype('int32'),
        X_test = lasagne.utils.floatX(X_test),
        Y_test = Y_test.astype('int32'),)
    
    
def load_data(dataset='cifar10'):
    if dataset == 'cifar10':
        return load_cifar10()
    elif dataset == 'cifar100':
        return load_cifar100()
    elif dataset == 'stl10':
        return load_stl10()
    else:
        raise NotImplementedError('dataset not found!')
    
def iterate_minibatches(inputs, targets, batchsize, shuffle=True, flip=False, crop=False, num=None, **kwargs):
    assert len(inputs) == len(targets)
    if not num:
        indices = np.arange(len(inputs))
    else:
        indices = np.arange(num)
    if shuffle:
        np.random.shuffle(indices)
    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        if crop:
            padded = np.pad(inputs[excerpt],((0,0),(0,0),(4,4),(4,4)),mode='constant')
            random_cropped = np.zeros(inputs[excerpt].shape, dtype=np.float32)
            crops = np.random.random_integers(0,high=8,size=(batchsize,2))
            for r in range(batchsize):
                random_cropped[r,:,:,:] = padded[r,:,crops[r,0]:(crops[r,0]+32),crops[r,1]:(crops[r,1]+32)]
            inp_exc = random_cropped
        else:
            inp_exc = inputs[excerpt]
        if flip:
            if np.random.randn(1)[0] > 0:
                inp_exc = inp_exc[:, :, :, ::-1]

        yield inp_exc, targets[excerpt]

def get_iterator(dataset):
    return iterate_minibatches