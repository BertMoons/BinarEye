# Copyright 2016 Bert Moons

# This file is part of BinarySense.

# BinarySense is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinarySense is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryNet.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import time
import gc

import numpy as np
np.random.seed(1234) # for reproducibility?

# specifying the gpu to use
#import theano.sandbox.cuda
#theano.sandbox.cuda.use('gpu3') 
#import os
#os.environ["THEANO_FLAGS"] = "cuda.root=/esat/leda1/cuda,device=gpu,floatX=float32"
import theano
import theano.tensor as T

import lasagne

import cPickle as pickle
import gzip

import binary_net

from collections import OrderedDict


def build_net(input,binary,stochastic=False,H=1.0,W_LR_scale="Glorot",activation=binary_net.binary_tanh_unit,epsilon=1e-4,alpha=.1,patch_size=32,channels=3,num_filters=256):

    cnn = lasagne.layers.InputLayer(
            shape=(None, channels, patch_size, patch_size),
            input_var=input)
#1        
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
 
    print(cnn.output_shape)
    
#2        
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 
 
    print(cnn.output_shape)
    
 
#3           
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    print(cnn.output_shape)
 
#4 
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    print(cnn.output_shape)
 
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print(cnn.output_shape)
 
#5                  
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)

    print(cnn.output_shape)
 
#6              
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation) 

    print(cnn.output_shape)
 
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print(cnn.output_shape)
 
#7                  
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
 
    print(cnn.output_shape)
 
#8
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=num_filters, 
            filter_size=(2, 2),
            pad='valid',
            nonlinearity=lasagne.nonlinearities.identity)
    
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)
                
    cnn = lasagne.layers.NonlinearityLayer(
            cnn,
            nonlinearity=activation)
 
    print(cnn.output_shape)
 
    cnn = binary_net.DenseLayer(
                cnn, 
                binary=binary,
                stochastic=stochastic,
                H=H,
                W_LR_scale=W_LR_scale,
                nonlinearity=lasagne.nonlinearities.identity,
                num_units=2)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)    

    return cnn
