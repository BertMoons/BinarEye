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
import argparse

import numpy as np
np.random.seed(1234) # for reproducibility?
import os
os.system("hostname")
import glob
listing = glob.glob('/usr/local/cuda*')
print(listing[0])
#os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/test/:%s/lib64/'%(listing[0])
#os.environ['CPATH'] = '/users/micas/bmoons/software/CUDNN/test/'


if 'LD_LIBRARY_PATH' not in os.environ:
    #os.environ['CPATH'] = '/users/micas/bmoons/software/CUDNN/test2/include/'
    os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/test2/lib64/:%s/lib64/'%(listing[0])
if 'PYLEARN2_DATA_PATH' not in os.environ:
    os.environ['PYLEARN2_DATA_PATH'] = '/esat/leda1/users/bmoons/PYLEARN2'
    print ('/esat/leda1/users/bmoons/PYLEARN2')


inc = '/users/micas/bmoons/software/CUDNN/test2/include'
lib64 = '/users/micas/bmoons/software/CUDNN/test2/lib64'

if listing:
    if(os.system("hostname")=='oculus.esat.kuleuven.be'):
        os.environ["THEANO_FLAGS"] = "cuda.root=%s,device=cuda0,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile
    else:
        os.environ["THEANO_FLAGS"] = "cuda.root=%s,device=gpu0,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile,lib.cnmem=0.925


print('path = ' +  os.environ['PATH'])
print('ld_library_path =  ' + os.environ['LD_LIBRARY_PATH'])
#print('cpath =  ' + os.environ['CPATH'])
print('theano flags =  ' + os.environ['THEANO_FLAGS'])


import lasagne
from theano import function, config, shared
import theano.tensor as T
import numpy
import theano.sandbox.cuda


import cPickle as pickle
import gzip

import binary_net
import build_net

from collections import OrderedDict

parser = argparse.ArgumentParser(description='run training on facedetection dataset')
parser.add_argument('-f','--filters',help='number of filters, typically 64 or 256', required=True, type=int)
args = parser.parse_args()

num_filters = args.filters
channels = 3
patch_size = 32

if __name__ == "__main__":
    # script save_raw
    save_raw = 0   
 
    # BN parameters
    batch_size = 128
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryConnect    
    binary = True
    print("binary = "+str(binary))
    stochastic = False
    print("stochastic = "+str(stochastic))
    # (-H,+H) are the two binary values
    # H = "Glorot"
    H = 1.
    print("H = "+str(H))
    # W_LR_scale = 1.    
    W_LR_scale = "Glorot" # "Glorot" means we are using the coefficients from Glorot's paper
    print("W_LR_scale = "+str(W_LR_scale))

    # BinaryOut
    if binary==True:
    	activation = binary_net.binary_tanh_unit
    	print("activation = binary_net.binary_tanh_unit")
    else:
		activation = T.nnet.relu
    
    # Training parameters
    num_epochs = 100
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    LR_ = [(0,.1), (5,10), (50,.1), (75,.1)]
    LR_decay = 1-1e-4
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    run_name = "OwnerSense_%s"%(str(num_filters))
    save_path = run_name + ".npz"
    print("save_path = "+str(save_path))

    load_path = "BinarySense.npz"
    print("load_path = "+str(load_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    val_directory = '/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/val'
    train_directory = '/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/train'
    
    print('Loading BinarySense Face Detection dataset...')

    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = build_net.build_net(input,binary=binary,stochastic=stochastic,H=H,W_LR_scale=W_LR_scale,activation=activation,epsilon=epsilon,alpha=alpha,patch_size=patch_size,channels=channels,num_filters=num_filters)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    fp_multiplier = 1
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output))*np.asarray([fp_multiplier,1]))
    err = T.mean(T.neq(T.argmax(train_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    train_1_when_0 = T.sum(T.gt(T.argmax(train_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX) # face =1, bg = 0 : fp
    train_0_when_1 = T.sum(T.lt(T.argmax(train_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX) # fn
    train_1_when_1 = T.sum(T.and_(T.argmax(train_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX)
    train_precision = train_1_when_1 / (train_1_when_1 + train_1_when_0) # TP/(TP+FP)
    train_recall = train_1_when_1 / (train_1_when_1 + train_0_when_1) # TP/(TP+FN)
    
    if binary:
        
        # W updates
        W = lasagne.layers.get_all_params(cnn, binary=True)
        W_grads = binary_net.compute_grads(loss,cnn)
        updates = lasagne.updates.adam(loss_or_grads=W_grads, params=W, learning_rate=LR)
        updates = binary_net.clipping_scaling(updates,cnn)
        
        # other parameters updates
        params = lasagne.layers.get_all_params(cnn, trainable=True, binary=False)
        updates = OrderedDict(updates.items() + lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR).items())
        
    else:
        params = lasagne.layers.get_all_params(cnn, trainable=True)
        updates = lasagne.updates.adam(loss_or_grads=loss, params=params, learning_rate=LR)

    test_output = lasagne.layers.get_output(cnn, deterministic=True)
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output))*np.asarray([fp_multiplier,1]))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    test_1_when_0 = T.sum(T.gt(T.argmax(test_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX) # face =1, bg = 0 : fp
    test_0_when_1 = T.sum(T.lt(T.argmax(test_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX) # fn
    test_1_when_1 = T.sum(T.and_(T.argmax(test_output, axis=1),T.argmax(target, axis=1)),dtype=theano.config.floatX)
    test_precision = test_1_when_1 / (test_1_when_1 + test_1_when_0) # TP/(TP+FP)
    test_recall = test_1_when_1 / (test_1_when_1 + test_0_when_1) # TP/(TP+FN)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], [loss,err,train_1_when_0,train_0_when_1, train_precision, train_recall], updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err, test_1_when_0,test_0_when_1,test_1_when_1])


    print('Training...')
    binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start, LR_, LR_decay,num_filters,run_name,val_directory,train_directory,
            num_epochs,
	    	save_path=save_path,
            shuffle_parts=shuffle_parts)

    '''
    with np.load(load_path) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(cnn, param_values)
    '''

