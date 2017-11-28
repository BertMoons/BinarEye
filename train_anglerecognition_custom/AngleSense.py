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
    os.environ['LD_LIBRARY_PATH'] = '/users/micas/bmoons/software/CUDNN/test2/lib64/:%s/lib64/'%(listing[0])
if 'PYLEARN2_DATA_PATH' not in os.environ:
    os.environ['PYLEARN2_DATA_PATH'] = '/esat/leda1/users/bmoons/PYLEARN2'
    print ('/esat/leda1/users/bmoons/PYLEARN2')


inc = '/users/micas/bmoons/software/CUDNN/test2/include'
lib64 = '/users/micas/bmoons/software/CUDNN/test2/lib64'

if listing:
    if(os.system("hostname")=='oculus.esat.kuleuven.be'):
        os.environ["THEANO_FLAGS"] = "cuda.root=%s,lib.cnmem=0.2,device=cuda0,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile
    else:
        os.environ["THEANO_FLAGS"] = "cuda.root=%s,lib.cnmem=0.2,device=gpu0,floatX=float32"%(listing[0])#,exception_verbosity=high,optimizer=fast_compile,lib.cnmem=0.925


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
parser.add_argument('-c','--classes',help='number of  classes, can be 3 or 7', required=True, type=int)
args = parser.parse_args()


num_filters = args.filters
num_classes = args.classes

run_name = 'anglerec_' + str(num_filters) + "_" + str(num_classes)

if __name__ == "__main__":
    # script save_raw
    save_raw = 0   
 
    # BN parameters
    batch_size = 100
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
	print("activation = T.nnet.relu")
    # activation = binary_net.binary_sigmoid_unit
    # print("activation = binary_net.binary_sigmoid_unit")
    
    # Training parameters
    num_epochs = 75
    print("num_epochs = "+str(num_epochs))
    
    # Decaying LR 
    LR_start = 0.001
    LR_ = [(0,1), (10,10), (30,.1), (60,.1)]
    LR_decay = 1-1e-4
    print("LR_decay = "+str(LR_decay))
    # BTW, LR decay might good for the BN moving average...

    save_path = run_name+".npz"
    print("save_path = "+str(save_path))

    load_path = "BinarySense.npz"
    print("load_path = "+str(load_path))
    
    shuffle_parts = 1
    print("shuffle_parts = "+str(shuffle_parts))
    
    print('Loading BinarySense Angle Recognition dataset...')


    basepath = '/volume1/users/bmoons/CUSTOM_ANGLE_RECOGNITION/numpy/'
    train_set_X = np.load(basepath + 'train_set_x_{}.npy'.format(num_classes))
    valid_set_X = np.load(basepath + 'valid_set_x_{}.npy'.format(num_classes))
    test_set_X = np.load(basepath + 'valid_set_x_{}.npy'.format(num_classes))

    train_set_Y = np.load(basepath + 'train_set_y_{}.npy'.format(num_classes))
    valid_set_Y = np.load(basepath + 'valid_set_y_{}.npy'.format(num_classes))
    test_set_Y = np.load(basepath + 'valid_set_y_{}.npy'.format(num_classes)) 


    # bc01 format
    # Inputs in the range [-1,+1]
    train_set_X = np.reshape(np.subtract(np.multiply(2./255.,train_set_X),1.),(-1,3,32,32))
    valid_set_X = np.reshape(np.subtract(np.multiply(2./255.,valid_set_X),1.),(-1,3,32,32))
    test_set_X = np.reshape(np.subtract(np.multiply(2./255.,test_set_X),1.),(-1,3,32,32))

    # Number of feature maps
    num_maps = 85 # 256/3 input channels

    # Quantize
    s = train_set_X / np.abs(train_set_X)
    train_set_X=(2*(s*np.ceil(np.abs(train_set_X)*num_maps/2))-s*1).astype('float32')
    # Quantize
    s = valid_set_X / np.abs(valid_set_X)
    valid_set_X=(2*(s*np.ceil(np.abs(valid_set_X)*num_maps/2))-s*1).astype('float32')
    # Quantize
    s = test_set_X / np.abs(test_set_X)
    test_set_X=(2*(s*np.ceil(np.abs(test_set_X)*num_maps/2))-s*1).astype('float32')
    
    # Onehot the targets
    train_set_Y = np.float32(np.eye(num_classes)[train_set_Y.astype(np.int)])    
    valid_set_Y = np.float32(np.eye(num_classes)[valid_set_Y.astype(np.int)])
    test_set_Y = np.float32(np.eye(num_classes)[test_set_Y.astype(np.int)])
    
    # for hinge loss
    train_set_Y = 2* train_set_Y - 1.
    valid_set_Y = 2* valid_set_Y - 1.
    test_set_Y = 2* test_set_Y - 1.  



    print('Building the CNN...') 
    
    # Prepare Theano variables for inputs and targets
    input = T.tensor4('inputs')
    target = T.matrix('targets')
    LR = T.scalar('LR', dtype=theano.config.floatX)

    cnn = build_net.build_net(input,binary=binary,stochastic=stochastic,H=H,W_LR_scale=W_LR_scale,activation=activation,epsilon=epsilon,alpha=alpha,num_filters=num_filters,num_classes=num_classes)

    train_output = lasagne.layers.get_output(cnn, deterministic=False)
    print('CNN built')     

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
    val_fn = theano.function([input, target], [test_loss, test_err, test_output])


    print('Training...')
    binary_net.train(
            train_fn,val_fn,
            cnn,
            batch_size,
            LR_start,LR_,LR_decay, run_name,
            num_epochs,
            train_set_X,train_set_Y,
            valid_set_X,valid_set_Y,
            test_set_X,test_set_Y,
	        save_path=save_path,
            shuffle_parts=shuffle_parts)

