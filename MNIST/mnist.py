# Copyright 2016 Matthieu Courbariaux

# This file is part of BinaryNet.

# BinaryNet is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# BinaryNet is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with BinaryNet.  If not, see <http://www.gnu.org/licenses/>.

from __future__ import print_function

import sys
import os
import time
import gc

import numpy as np
np.random.seed(1234) # for reproducibility?

import glob
listing = glob.glob('/usr/local/cuda*')
print(listing[0])
#os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/test/:%s/lib64/'%(listing[0])
#os.environ['CPATH'] = '/users/micas/bmoons/software/CUDNN/test/'


if 'LD_LIBRARY_PATH' not in os.environ:
    os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/test2/lib64/:%s/lib64/'%(listing[0])
if 'PYLEARN2_DATA_PATH' not in os.environ:
    os.environ['PYLEARN2_DATA_PATH'] = '/esat/leda1/users/bmoons/PYLEARN2'
    print ('/esat/leda1/users/bmoons/PYLEARN2')

dst = '%s/include/cudnn.h'%(listing[0])


#os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/cuda/lib64/:$LD_LIBRARY_PATH'
#os.environ['PATH']='/users/micas/bmoons/software/CUDNN/cuda/include/:$PATH'

print('path = ' +  os.environ['PATH'])
print('ld_library_path =  ' + os.environ['LD_LIBRARY_PATH'])


if listing:
    if(os.system("hostname")=='oculus.esat.kuleuven.be'):
	    os.environ["THEANO_FLAGS"] = "cuda.root=%s,device=cuda0,lib.cnmem=0.875,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile
    else:
	    os.environ["THEANO_FLAGS"] = "cuda.root=%s,device=gpu0,lib.cnmem=0.875,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile

import lasagne
from theano import function, config, shared
import theano.tensor as T
import numpy
import theano.sandbox.cuda


import theano

import cPickle as pickle
import gzip

import binary_net

from pylearn2.datasets.zca_dataset import ZCA_Dataset   
from pylearn2.datasets.cifar10 import CIFAR10 
from pylearn2.datasets.mnist import MNIST
from pylearn2.utils import serial

from collections import OrderedDict

parser = argparse.ArgumentParser(description='run training on facedetection dataset')
parser.add_argument('-f','--filters',help='number of filters, typically 64 or 256', required=True, type=int)
args = parser.parse_args()

num_filters = args.filters

if __name__ == "__main__":
    


    filters = num_filters;
    # BN parameters
    batch_size = 50
    print("batch_size = "+str(batch_size))
    # alpha is the exponential moving average factor
    alpha = .1
    print("alpha = "+str(alpha))
    epsilon = 1e-4
    print("epsilon = "+str(epsilon))
    
    # BinaryOut
    activation = binary_net.binary_tanh_unit
    print("activation = binary_net.binary_tanh_unit")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
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
    
    # Training parameters
    num_epochs = 150
    print("num_epochs = "+str(num_epochs))
	
	# load or run
    train = True
    
    # Decaying LR 
    LR_start = 0.001
    print("LR_start = "+str(LR_start))
    LR_fin = 0.00001
    print("LR_fin = "+str(LR_fin))
    #LR_decay = (LR_fin/LR_start)**(1./num_epochs)
    LR_decay = 1-1e-1
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    save_path = "cifar10_21linput_nopadding_augmented_64ch.npz"
    print("save_path = "+str(save_path))

    load_path = save_path
    print("load_path = "+str(load_path))
    
    train_set_size = 45000
    print("train_set_size = "+str(train_set_size))
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading MNIST dataset...')

    train_set_size = 60000
    train_set = MNIST(which_set="train",start=0,stop = train_set_size)
    #    valid_set = MNIST(which_set="train",start=train_set_size,stop = 60000)
    valid_set = MNIST(which_set="test")
    test_set = MNIST(which_set="test")
    
    #train_set.X = np.reshape(np.subtract(np.multiply(2.,train_set.X),1.),(-1,1,28,28))
    train_set.X = np.reshape(train_set.X,(-1,1,28,28))
    valid_set.X = np.reshape(np.subtract(np.multiply(2.,valid_set.X),1.),(-1,1,28,28))
    test_set.X = np.reshape(np.subtract(np.multiply(2.,test_set.X),1.),(-1,1,28,28))

    # Number of feature maps
    num_maps = (256/(1*256/filters)) # 256/12 input channels


    # Quantize
    #s = train_set.X / np.abs(train_set.X)
    #train_set.X=(2*(s*np.ceil(np.abs(train_set.X)*num_maps/2))-s*1).astype('float32')
    # Quantize
    s = valid_set.X / np.abs(valid_set.X)
    valid_set.X=(2*(s*np.ceil(np.abs(valid_set.X)*num_maps/2))-s*1).astype('float32')
    # Quantize
    s = test_set.X / np.abs(test_set.X)
    test_set.X=(2*(s*np.ceil(np.abs(test_set.X)*num_maps/2))-s*1).astype('float32')



    # flatten targets
    train_set.y = np.hstack(train_set.y)
    valid_set.y = np.hstack(valid_set.y)
    test_set.y = np.hstack(test_set.y)

    # Onehot the targets
    train_set.y = np.float32(np.eye(10)[train_set.y])    
    valid_set.y = np.float32(np.eye(10)[valid_set.y])
    test_set.y = np.float32(np.eye(10)[test_set.y])

    # for hinge loss
    train_set.y = 2* train_set.y - 1.
    valid_set.y = 2* valid_set.y - 1.
    test_set.y = 2* test_set.y - 1.
    # enlarge train data set by mirrroring
    x_train_flip = train_set.X[:,:,:,::-1]
    y_train_flip = train_set.y
    train_set.X = np.concatenate((train_set.X,x_train_flip),axis=0)
    train_set.y = np.concatenate((train_set.y,y_train_flip),axis=0)
    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = lasagne.layers.InputLayer(
            shape=(None, 1, 28, 28),
            input_var=input)
#1        
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=filters, 
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

    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))
 
    print(cnn.output_shape)

 
#2 
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=filters, 
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
 
    cnn = lasagne.layers.MaxPool2DLayer(cnn, pool_size=(2, 2))

    print(cnn.output_shape)
 
#3                  
    cnn = binary_net.Conv2DLayer(
            cnn, 
            binary=binary,
            stochastic=stochastic,
            H=H,
            W_LR_scale=W_LR_scale,
            num_filters=filters, 
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
            num_filters=filters, 
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
                num_units=10)      
                  
    cnn = lasagne.layers.BatchNormLayer(
            cnn,
            epsilon=epsilon, 
            alpha=alpha)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    
    # squared hinge loss
    loss = T.mean(T.sqr(T.maximum(0.,1.-target*train_output)))
    
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
    test_loss = T.mean(T.sqr(T.maximum(0.,1.-target*test_output)))
    test_err = T.mean(T.neq(T.argmax(test_output, axis=1), T.argmax(target, axis=1)),dtype=theano.config.floatX)
    
    # Compile a function performing a training step on a mini-batch (by giving the updates dictionary) 
    # and returning the corresponding training loss:
    train_fn = theano.function([input, target, LR], loss, updates=updates)

    # Compile a second function computing the validation loss and accuracy:
    val_fn = theano.function([input, target], [test_loss, test_err])
    
    if train:
		print('Training...')
		binary_net.train(
				train_fn,val_fn,
				cnn,
				batch_size,
				LR_start,LR_decay,
				num_epochs,
				train_set.X,train_set.y,
				valid_set.X,valid_set.y,
				test_set.X,test_set.y,
			save_path=save_path,
				shuffle_parts=shuffle_parts)
