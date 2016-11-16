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
from stochastic_pool import stochastic_max_pool_bc01, weighted_max_pool_bc01

from lib.ops import deconv

class DeconvLayer(lasagne.layers.conv.BaseConvLayer):
    def __init__(self, incoming, num_filters, filter_size, stride=(1, 1),
                     crop=0, untie_biases=False,
                     W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.),
                     nonlinearity=lasagne.nonlinearities.rectify, flip_filters=False, **kwargs):
        super(DeconvLayer, self).__init__(incoming, num_filters, filter_size, stride, crop,
                                          untie_biases, W, b, nonlinearity, flip_filters,
                                          n=2, **kwargs)

    def get_W_shape(self):
        num_input_channels = self.input_shape[1]
        return (num_input_channels, self.num_filters) + self.filter_size

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.num_filters, 2*input_shape[2], 2*input_shape[3])

    def convolve(self, input, **kwargs):
        return deconv(input, self.W, subsample=(2, 2), border_mode='half')


class StochasticPool2DLayer(lasagne.layers.Layer):
    def __init__(self, incoming, pool_size=2, maxpool=True, grid_size=None, **kwargs):
        super(StochasticPool2DLayer, self).__init__(incoming, **kwargs)
        self.rng = T.shared_randomstreams.RandomStreams(123)
        self.pool_size = pool_size
        self.maxpool = maxpool
        if grid_size:
            self.grid_size = grid_size
        else:
            self.grid_size = pool_size
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],
                input_shape[2]/self.pool_size, input_shape[3]/self.pool_size)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        if self.maxpool:
            input = T.signal.pool.pool_2d(input,
                                          ds=(self.pool_size,)*2,
                                          ignore_border=True,
                                          st=(1,1),
                                          mode='max')
        if deterministic:
            input = T.signal.pool.pool_2d(input,
                                          ds=(self.pool_size,)*2,
                                          ignore_border=True,
                                          st=(self.pool_size,)*2,
                                          padding=(self.pool_size/2,)*2,
                                          mode='average_exc_pad')

            return input
            # return input[:, :, ::self.pool_size, ::self.pool_size]
                
        else:
            w, h = self.input_shape[2:]
            n_w, n_h = w / self.grid_size, h / self.grid_size
            n_sample_per_grid = self.grid_size / self.pool_size
            idx_w = []
            idx_h = []
            
            for i in range(n_w):
                offset = self.grid_size * i
                if i < n_w - 1:
                    this_n = self.grid_size
                else:
                    this_n = input.shape[2] - offset
                this_idx = T.sort(self.rng.permutation(size=(1,), n=this_n)[0, :n_sample_per_grid])
                idx_w.append(offset + this_idx)

            for i in range(n_h):
                offset = self.grid_size * i
                if i < n_h - 1:
                    this_n = self.grid_size
                else:
                    this_n = input.shape[3] - offset
                this_idx = T.sort(self.rng.permutation(size=(1,), n=this_n)[0, :n_sample_per_grid])
                idx_h.append(offset + this_idx)
            idx_w = T.concatenate(idx_w, axis=0)
            idx_h = T.concatenate(idx_h, axis=0)

            output = input[:, :, idx_w][:, :, :, idx_h]
            
            return output

class ZeilerPool2DLayer(Layer):
    def __init__(self, incoming, pool_size=2, **kwargs):
        super(ZeilerPool2DLayer, self).__init__(incoming, **kwargs)
        self.pool_size = pool_size
        
    def get_output_shape_for(self, input_shape):
        return (input_shape[0], input_shape[1],
                input_shape[2]/self.pool_size, input_shape[3]/self.pool_size)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        if deterministic:
            pool_fn = weighted_max_pool_bc01
        else:
            pool_fn = stochastic_max_pool_bc01
        
        return  pool_fn(input, (self.pool_size,)*2, (self.pool_size,)*2, self.input_shape[2:])

            
def build_nin(input_var, option, ny=10, visualize=False, **kwargs):
    if option.startswith('stochastic'):
        grid_sizes = [int(s) for s in option.split('-')[1:]]
        option = option.split('-')[0]
    net = InputLayer((None, 3, 32, 32), input_var=input_var)
    net = batch_norm(Conv2DLayer(net, num_filters=192, filter_size=5, pad='same', flip_filters=False))
    net = batch_norm(NINLayer(net, num_units=160))    
    net = batch_norm(NINLayer(net, num_units=96))
    if option == 'standard':
        net = MaxPool2DLayer(net, pool_size=2)
        net = DropoutLayer(net, p=0.5)
    elif option == 'stochastic':
        net = StochasticPool2DLayer(net, pool_size=2, maxpool=True, grid_size=grid_sizes[0])
        inv_net1 = batch_norm(DeconvLayer(net, 128, (5,5)))
        inv_net1 = batch_norm(Conv2DLayer(inv_net1, 3, (5,5), pad='same', nonlinearity=None))
    elif option == 'zeiler':
        net = ZeilerPool2DLayer(net, pool_size=2)
    else:
        raise NotImplementedError

    net = batch_norm(Conv2DLayer(net, num_filters=192, filter_size=5, pad='same', flip_filters=False))
    net = batch_norm(NINLayer(net, num_units=192))
    net = batch_norm(NINLayer(net, num_units=192))
    if option == 'standard':
        net = MaxPool2DLayer(net, pool_size=2)
        net = DropoutLayer(net, p=0.5)
    elif option == 'stochastic':
        net = StochasticPool2DLayer(net, pool_size=2, maxpool=True, grid_size=grid_sizes[1])
        inv_net2 = batch_norm(DeconvLayer(net, 128, (5,5)))
        inv_net2 = batch_norm(Conv2DLayer(inv_net2, 128, (5, 5), pad='same'))
        inv_net2 = batch_norm(DeconvLayer(inv_net2, 128, (5, 5)))
        inv_net2 = batch_norm(Conv2DLayer(inv_net2, 3, (5, 5), pad='same', nonlinearity=None))

    elif option == 'zeiler':
        net = ZeilerPool2DLayer(net, pool_size=2)
    else:
        raise NotImplementedError

    net = batch_norm(Conv2DLayer(net, num_filters=192, filter_size=3, pad='same', flip_filters=False))
    net = batch_norm(NINLayer(net, num_units=192))
    net = batch_norm(DenseLayer(GlobalPoolLayer(net), num_units=ny, nonlinearity=T.nnet.softmax))

    if not visualize:
        return net
    else:
        return net, inv_net1, inv_net2

    
def build_resnet(input_var, option, n=3, ny=10, visualize=False, **kwargs):
    sys.setrecursionlimit(10000)
    
    if option.startswith('stochastic'):
        grid_sizes = [int(s) for s in option.split('-')[1:]]
        option = option.split('-')[0]

    def residual_block(l, increase_dim=False, down_sample=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
            
        if down_sample:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        stack_1 = batch_norm(Conv2DLayer(l, num_filters=out_num_filters,
                                         filter_size=(3,3), stride=first_stride,
                                         nonlinearity=T.nnet.relu, pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        stack_2 = batch_norm(Conv2DLayer(stack_1, num_filters=out_num_filters,
                                         filter_size=(3,3), stride=(1,1),
                                         nonlinearity=None, pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        
        if down_sample:
            l = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2],
                                lambda s: (s[0], s[1], s[2]//2, s[3]//2))
        if increase_dim:
            l = PadLayer(l, [out_num_filters//4,0,0], batch_ndim=1)
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=T.nnet.relu)
        
        return block

    l_in = InputLayer(shape=(None, 3, 32, 32), input_var=input_var)

    l = batch_norm(Conv2DLayer(l_in, num_filters=32,
                               filter_size=(3,3), stride=(1,1),
                               nonlinearity=T.nnet.relu, pad='same',
                               W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
    
    for _ in range(n):
        l = residual_block(l)
    ########################################################################################
        
    if option == 'standard':
        down_sample = True
    elif option == 'stochastic':
        l = StochasticPool2DLayer(l, pool_size=2, maxpool=True, grid_size=grid_sizes[0])
        inv_net1 = batch_norm(DeconvLayer(l, 128, (5,5)))
        inv_net1 = batch_norm(Conv2DLayer(inv_net1, 3, (5,5), pad='same', nonlinearity=None))

        down_sample = False
    elif option == 'zeiler':
        l = ZeilerPool2DLayer(l, pool_size=2)
        down_sample = False
    else:
        raise NotImplementedError

    l = residual_block(l, increase_dim=True, down_sample=down_sample)
    for _ in range(1,n):
        l = residual_block(l)
    ########################################################################################
        
    if option == 'standard':
        down_sample = True
    elif option == 'stochastic':
        l = StochasticPool2DLayer(l, pool_size=2, maxpool=True, grid_size=grid_sizes[1])
        inv_net2 = batch_norm(DeconvLayer(l, 128, (5,5)))
        inv_net2 = batch_norm(Conv2DLayer(inv_net2, 128, (5, 5), pad='same'))
        inv_net2 = batch_norm(DeconvLayer(inv_net2, 128, (5, 5)))
        inv_net2 = batch_norm(Conv2DLayer(inv_net2, 3, (5, 5), pad='same', nonlinearity=None))

        down_sample = False
    elif option == 'zeiler':
        l = ZeilerPool2DLayer(l, pool_size=2)
        down_sample = False
    else:
        raise NotImplementedError

    l = residual_block(l, increase_dim=True, down_sample=down_sample)
    for _ in range(1,n):
        l = residual_block(l)
    ########################################################################################
        
    l = GlobalPoolLayer(l)
    network = DenseLayer(l, num_units=ny, W=lasagne.init.HeNormal(), nonlinearity=T.nnet.softmax)

    if not visualize:
        return network
    else:
        return network, inv_net1, inv_net2


def build_stl10resnet(input_var, option, ny=10, **kwargs):
    sys.setrecursionlimit(10000)
    
    if option.startswith('stochastic'):
        grid_sizes = [int(s) for s in option.split('-')[1:]]
        option = option.split('-')[0]

    def residual_block(l, increase_dim=False, down_sample=False):
        input_num_filters = l.output_shape[1]
        if increase_dim:
            out_num_filters = input_num_filters*2
        else:
            out_num_filters = input_num_filters
            
        if down_sample:
            first_stride = (2,2)
        else:
            first_stride = (1,1)

        stack_1 = batch_norm(Conv2DLayer(l, num_filters=out_num_filters,
                                         filter_size=(3,3), stride=first_stride,
                                         nonlinearity=T.nnet.relu, pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        stack_2 = batch_norm(Conv2DLayer(stack_1, num_filters=out_num_filters,
                                         filter_size=(3,3), stride=(1,1),
                                         nonlinearity=None, pad='same',
                                         W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))
        
        # add shortcut connections
        
        if down_sample:
            l = ExpressionLayer(l, lambda X: X[:, :, ::2, ::2],
                                lambda s: (s[0], s[1], s[2]//2, s[3]//2))
        if increase_dim:
            l = PadLayer(l, [out_num_filters//4,0,0], batch_ndim=1)
        block = NonlinearityLayer(ElemwiseSumLayer([stack_2, l]),nonlinearity=T.nnet.relu)
        
        return block
        
    ########################################################################################
    l_in = InputLayer(shape=(None, 3, 96, 96), input_var=input_var)

    l = batch_norm(Conv2DLayer(l_in, num_filters=32,
                               filter_size=(5,5), stride=(1,1),
                               nonlinearity=T.nnet.relu, pad='same',
                               W=lasagne.init.HeNormal(gain='relu'), flip_filters=False))

    ########################################################################################
    for i in range(4):    
        if i == 0:
            increase_dim = False
        else:
            increase_dim = True
        if option == 'standard':
            down_sample = True
        elif option == 'stochastic':
            l = StochasticPool2DLayer(l, pool_size=2, maxpool=True, grid_size=grid_sizes[i])
            down_sample = False
        elif option == 'zeiler':
            l = ZeilerPool2DLayer(l, pool_size=2)
            down_sample = False
        else:
            raise NotImplementedError
        l = residual_block(l, increase_dim=increase_dim, down_sample=down_sample)
        l = residual_block(l)

    ########################################################################################
    
    l = GlobalPoolLayer(l)
    network = DenseLayer(l, num_units=ny, W=lasagne.init.HeNormal(), nonlinearity=T.nnet.softmax)

    return network


    
def build_net(option, arch, ny, visualize=False):
    # Prepare Theano variables for inputs and targets
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')

    # Create neural network model
    print "Building model and compiling functions..."
    if arch.startswith('resnet'):
        network = build_resnet(input_var, option, n=int(arch.split('-')[1]), ny=ny, visualize=visualize)
    elif arch == 'nin':
        network = build_nin(input_var, option, ny=ny, visualize=visualize)
    elif arch == 'stl10resnet':
        network = build_stl10resnet(input_var, option, ny=ny)

    return input_var, target_var, network