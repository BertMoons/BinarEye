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

import time

from collections import OrderedDict

import numpy as np

# specifying the gpu to use
# import theano.sandbox.cuda
# theano.sandbox.cuda.use('gpu1') 
import theano
import theano.tensor as T

import lasagne

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from theano.scalar.basic import UnaryScalarOp, same_out_nocomplex
from theano.tensor.elemwise import Elemwise

from tools.image import ImageDataGenerator, random_transform, standardize

import os, os.path

from augment import augment_images, adapt_to_binareye



# Our own rounding function, that does not set the gradient to 0 like Theano's
class Round3(UnaryScalarOp):
    
    def c_code(self, node, name, (x,), (z,), sub):
        return "%(z)s = round(%(x)s);" % locals()
    
    def grad(self, inputs, gout):
        (gz,) = gout
        return gz, 
        
round3_scalar = Round3(same_out_nocomplex, name='round3')
round3 = Elemwise(round3_scalar)

def hard_sigmoid(x):
    return T.clip((x+1.)/2.,0,1)

# The neurons' activations binarization function
# It behaves like the sign function during forward propagation
# And like:
#   hard_tanh(x) = 2*hard_sigmoid(x)-1 
# during back propagation
def binary_tanh_unit(x):
    return 2.*round3(hard_sigmoid(x))-1.
    
def binary_sigmoid_unit(x):
    return round3(hard_sigmoid(x))
    
# The weights' binarization function, 
# taken directly from the BinaryConnect github repository 
# (which was made available by his authors)
def binarization(W,H,binary=True,deterministic=False,stochastic=False,srng=None):
    
    # (deterministic == True) <-> test-time <-> inference-time
    if not binary or (deterministic and stochastic):
        # print("not binary")
        Wb = W
    
    else:
        
        # [-1,1] -> [0,1]
        Wb = hard_sigmoid(W/H)
        # Wb = T.clip(W/H,-1,1)
        
        # Stochastic BinaryConnect
        if stochastic:
        
            # print("stoch")
            Wb = T.cast(srng.binomial(n=1, p=Wb, size=T.shape(Wb)), theano.config.floatX)

        # Deterministic BinaryConnect (round to nearest)
        else:
            # print("det")
            Wb = T.round(Wb)
        
        # 0 or 1 -> -1 or 1
        Wb = T.cast(T.switch(Wb,H,-H), theano.config.floatX)
    
    return Wb

# This class extends the Lasagne DenseLayer to support BinaryConnect
class DenseLayer(lasagne.layers.DenseLayer):
    
    def __init__(self, incoming, num_units, 
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.H = np.float32(np.sqrt(1.5/ (num_inputs + num_units)))
            # print("H = "+str(self.H))
            
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(incoming.output_shape[1:]))
            self.W_LR_scale = np.float32(1./np.sqrt(1.5/ (num_inputs + num_units)))

        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        
        if self.binary:
            super(DenseLayer, self).__init__(incoming, num_units, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
            
        else:
            super(DenseLayer, self).__init__(incoming, num_units, **kwargs)
        
    def get_output_for(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(DenseLayer, self).get_output_for(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This class extends the Lasagne Conv2DLayer to support BinaryConnect
class Conv2DLayer(lasagne.layers.Conv2DLayer):
    
    def __init__(self, incoming, num_filters, filter_size,
        binary = True, stochastic = True, H=1.,W_LR_scale="Glorot", **kwargs):
        
        self.binary = binary
        self.stochastic = stochastic
        
        self.H = H
        if H == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.H = np.float32(np.sqrt(1.5 / (num_inputs + num_units)))
            # print("H = "+str(self.H))
        
        self.W_LR_scale = W_LR_scale
        if W_LR_scale == "Glorot":
            num_inputs = int(np.prod(filter_size)*incoming.output_shape[1])
            num_units = int(np.prod(filter_size)*num_filters) # theoretically, I should divide num_units by the pool_shape
            self.W_LR_scale = np.float32(1./np.sqrt(1.5 / (num_inputs + num_units)))
            # print("W_LR_scale = "+str(self.W_LR_scale))
            
        self._srng = RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
            
        if self.binary:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, W=lasagne.init.Uniform((-self.H,self.H)), **kwargs)   
            # add the binary tag to weights            
            self.params[self.W]=set(['binary'])
        else:
            super(Conv2DLayer, self).__init__(incoming, num_filters, filter_size, **kwargs)    
    
    def convolve(self, input, deterministic=False, **kwargs):
        
        self.Wb = binarization(self.W,self.H,self.binary,deterministic,self.stochastic,self._srng)
        Wr = self.W
        self.W = self.Wb
            
        rvalue = super(Conv2DLayer, self).convolve(input, **kwargs)
        
        self.W = Wr
        
        return rvalue

# This function computes the gradient of the binary weights
def compute_grads(loss,network):
        
    layers = lasagne.layers.get_all_layers(network)
    grads = []
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        if params:
            # print(params[0].name)
            grads.append(theano.grad(loss, wrt=layer.Wb))
                
    return grads

# This functions clips the weights after the parameter update
def clipping_scaling(updates,network):
    
    layers = lasagne.layers.get_all_layers(network)
    updates = OrderedDict(updates)
    
    for layer in layers:
    
        params = layer.get_params(binary=True)
        for param in params:
            print("W_LR_scale = "+str(layer.W_LR_scale))
            print("H = "+str(layer.H))
            updates[param] = param + layer.W_LR_scale*(updates[param] - param)
            updates[param] = T.clip(updates[param], -layer.H,layer.H)     

    return updates
        
# Given a dataset and a model, this function trains the model on the dataset for several epochs
# (There is no default trainer function in Lasagne yet)
def train(train_fn,val_fn,
            model,
            batch_size,
            LR, LR_, LR_decay,num_filters,run_name, val_directory, train_directory,
            num_epochs,
            save_path=None,
            shuffle_parts=1,
            seed=0):

    datagen_tr = ImageDataGenerator(
        samplewise_center = False,
        samplewise_std_normalization=False,
        rotation_range=45,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        rescale = 1./255)

    datagen_val = ImageDataGenerator(
        rescale = 1./255)

    def random_crop(x,random_crop_size=(32,32),sync_seed=1, **kwargs):
        np.random.seed(sync_seed)
        w,h = x.shape[1], x.shape[2]
        rangew = (w-random_crop_size[0]) // 2
        rangeh = (h-random_crop_size[1]) // 2
        offsetw = o if rangew==0 else np.random.randint(rangew)
        offseth = o if rangeh==0 else np.random.randint(rangeh)
        return x[:,offsetw:offsetw+random_crop_size[0], offseth:offseth+random_crop_size[1]]    

    def center_crop(x,center_crop_size=(32,32),**kwargs):
        centerw,centerh=x.shape[1]//2,x.shape[2]//2
        halfw,halfh = center_crop_size[0]//2, center_crop_size[1]//2
        return x[:,centerw-halfw:centerw+halfw, centerh-halfh:centerh+halfh]

    def make_chip_compatible(x,num_maps=86, **kwargs):
        x = np.subtract(np.multiply(x,2),1)
        s = x / np.abs(x)
        s[np.isnan(s)]=1
        y=(2*(s*np.ceil(np.abs(x)*num_maps/2))-s*1).astype('float32')
        return y  

    datagen_tr.config['random_crop_size'] = (32,32)
    datagen_tr.config['num_maps'] = np.floor(256./(3*(256/num_filters))) # 256/3 input channels
    datagen_tr.set_pipeline([standardize,random_transform,make_chip_compatible])

    datagen_val.config['random_crop_size'] = (32,32)
    datagen_val.config['center_crop_size'] = (32,32)
    datagen_val.config['num_maps'] = np.floor(256./(3*(256/num_filters))) # 256/3 input channels
    datagen_val.set_pipeline([random_transform,make_chip_compatible])

    train_loss_array = np.asarray([])
    train_err_array = np.asarray([])
    train_1w0_array = np.asarray([])
    train_0w1_array = np.asarray([])
    train_precision_array = np.asarray([])
    train_recall_array = np.asarray([])



    val_loss_array = np.asarray([])
    val_err_array = np.asarray([])
    val_1w0_array = np.asarray([])
    val_0w1_array = np.asarray([])
    val_precision_array = np.asarray([])
    val_recall_array = np.asarray([])
    
    # This function trains the model a full epoch (on the whole dataset)
    def train_epoch(train_x,train_y, train_loss_array, train_err_array, train_1w0_array,train_0w1_array,train_precision_array,train_recall_array,LR, LR_decay,directory):
        
        loss = 0
        batch=0
        nr_batches = train_y.shape[0]/batch_size
        minibatch_start_time = time.time()
        np.save('./train_val_data/batches_'+run_name+'.npy',nr_batches)
        #for x_batch, y_batch in datagen_tr.flow_from_directory(directory,target_size=(32,32),batch_size=batch_size,class_mode='categorical'):
        #for x_batch, y_batch in datagen_tr.flow(train_x,train_y,batch_size=batch_size):
        for i in range(nr_batches):
	    train_x_ = augment_images(train_x[i*batch_size:(i+1)*batch_size],rotation_range=30,height_shift_range=0.1, width_shift_range=0.1,shear_range=0.1,zoom_range=(1,1))
	    train_x_ = adapt_to_binareye(train_x_.transpose([0,3,1,2]),filters=num_filters)
            new_loss, new_err, new_1w0, new_0w1, new_precision,new_recall = train_fn(train_x_,np.subtract(np.multiply(train_y[i*batch_size:(i+1)*batch_size],2),1),LR)
            loss += new_loss
            train_loss_array = np.append(train_loss_array,new_loss)
            np.save('./train_val_data/train_loss_'+run_name+'.npy',train_loss_array)
            train_err_array = np.append(train_err_array,new_err)
            np.save('./train_val_data/train_err_'+run_name+'.npy',train_err_array)
            train_1w0_array = np.append(train_1w0_array,new_1w0)
            np.save('./train_val_data/train_1w0_'+run_name+'.npy',train_1w0_array)
            train_0w1_array = np.append(train_0w1_array,new_0w1)
            np.save('./train_val_data/train_0w1_'+run_name+'.npy',train_0w1_array)
            train_precision_array = np.append(train_precision_array,new_precision)
            np.save('./train_val_data/train_precision_'+run_name+'.npy',train_precision_array)
            train_recall_array = np.append(train_recall_array,new_recall)
            np.save('./train_val_data/train_recall_'+run_name+'.npy',train_recall_array)
            if(batch%10==0):
                minibatch_duration = time.time() - minibatch_start_time
                print('batch {:4d} / {} | loss: {:4f} | error: {} | precision: {} | recall: {} | LR = {} | time = {}'.format(batch,nr_batches,new_loss+0, new_err, new_precision, new_recall,LR,minibatch_duration))
                minibatch_start_time = time.time()
            batch += 1
            LR *=LR_decay
            if batch >= nr_batches:
                break
        
        loss/=nr_batches
        
        return loss, LR, train_loss_array, train_err_array, train_1w0_array,train_0w1_array,train_precision_array,train_recall_array
    
    # This function tests the model a full epoch (on the whole dataset)
    def val_epoch(val_x,val_y):
        
        err = 0
        loss = 0
        _1w0 = 0
        _0w1 = 0
        _1w1 = 0
        batch=0
        #X = sum([len(files) for r, d, files in os.walk(directory)])
        nr_batches = val_y.shape[0]/batch_size
        #for x_batch, y_batch in datagen_val.flow_from_directory(directory,target_size=(32,32),batch_size=batch_size,class_mode='categorical'):
        for i in range(nr_batches):
	    val_x_ = val_x[i*batch_size:(i+1)*batch_size].transpose([0,3,1,2])
            val_x_ = adapt_to_binareye(val_x_,filters=num_filters)
            new_loss, new_err, new_1w0, new_0w1, new_1w1 = val_fn(val_x_,np.subtract(np.multiply(val_y[i*batch_size:(i+1)*batch_size],2),1))
            err += new_err
            loss += new_loss
            _1w0 += new_1w0
            _0w1 += new_0w1
            _1w1 += new_1w1
            batch += 1
            if batch >= nr_batches:
                break
        
        err = err / nr_batches * 100
        loss /= nr_batches
        _1w0 = _1w0 
        _0w1 = _0w1 

        return err, loss, _1w0, _0w1, _1w1

    # A function which shuffles a dataset
    def shuffle(X,y):
        
        # print(len(X))
        
        chunk_size = len(X)/shuffle_parts
        shuffled_range = range(chunk_size)
        
        X_buffer = np.copy(X[0:chunk_size])
        y_buffer = np.copy(y[0:chunk_size])
        
        for k in range(shuffle_parts):
            
            np.random.shuffle(shuffled_range)

            for i in range(chunk_size):
                
                X_buffer[i] = X[k*chunk_size+shuffled_range[i]]
                y_buffer[i] = y[k*chunk_size+shuffled_range[i]]
            
            X[k*chunk_size:(k+1)*chunk_size] = X_buffer
            y[k*chunk_size:(k+1)*chunk_size] = y_buffer

        return X,y
    
    # shuffle the train set
    best_train_loss = 100
    best_epoch = 1

    #train_x = np.reshape(np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/train_set_x.npy'),(-1,3,32,32)).astype('float32')
    train_x = np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/train_set_x.npy').astype('float32')
    train_y = np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/train_set_y.npy').astype('float32')
    # Image.fromarray(y_[0].astype(np.uint8),'RGB').save('test.png')

    #val_x = np.reshape(np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/valid_set_x.npy'),(-1,3,32,32)).astype('float32') 
    val_x = np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/valid_set_x.npy').astype('float32')
    val_y = np.load('/volume1/users/bmoons/CUSTOM_OWNER_RECOGNITION/numpy/valid_set_y.npy').astype('float32')

    train_y = np.float32(np.eye(2)[train_y.astype(np.int)])    
    val_y = np.float32(np.eye(2)[val_y.astype(np.int)])

    # enlarge train data set by mirroring
    #x_train_flip = train_x[:,:,:,::-1]
    x_train_flip = train_x[:,:,::-1,:]
    y_train_flip = train_y
    train_x = np.concatenate((train_x,x_train_flip),axis=0)
    train_y = np.concatenate((train_y,y_train_flip),axis=0)

    train_x, train_y = shuffle(train_x,train_y)
    
    # We iterate over epochs:
    for epoch in range(num_epochs):
        for ep,scale in LR_:
            if(epoch==ep):
                LR*=scale
        
        start_time = time.time()
        train_loss, LR, train_loss_array, train_err_array, train_1w0_array,train_0w1_array,train_precision_array,train_recall_array = train_epoch(train_x,train_y,train_loss_array,train_err_array,train_1w0_array,train_0w1_array, train_precision_array,train_recall_array,LR, LR_decay,train_directory)
        
        val_err, val_loss, val_1w0, val_0w1, val_1w1  = val_epoch(val_x,val_y)
        val_precision = val_1w1 / (val_1w1 + val_1w0) # TP/(TP+FP)
        val_recall = val_1w1 / (val_1w1 + val_0w1) # TP/(TP+FN)

        val_loss_array = np.append(val_loss_array,val_loss)
        np.save('./train_val_data/val_loss_'+run_name+'.npy',val_loss_array)
        val_err_array = np.append(val_err_array,val_err)
        np.save('./train_val_data/val_err_'+run_name+'.npy',val_err_array)
        val_1w0_array = np.append(val_1w0_array,val_1w0)
        np.save('./train_val_data/val_1w0_'+run_name+'.npy',val_1w0_array)
        val_0w1_array = np.append(val_0w1_array,val_0w1)
        np.save('./train_val_data/val_0w1_'+run_name+'.npy',val_0w1_array)
        val_precision_array = np.append(val_precision_array,val_precision)
        np.save('./train_val_data/val_precision_'+run_name+'.npy',val_precision_array)
        val_recall_array = np.append(val_recall_array,val_recall)
        np.save('./train_val_data/val_recall_'+run_name+'.npy',val_recall_array)
        
        # test if validation error went down
        if train_loss <= best_train_loss:
            
            best_train_loss = train_loss
            best_1w0 = val_1w0
            best_0w1 = val_0w1
            best_epoch = epoch+1
            
            test_err, test_loss, test_1w0, test_0w1,test_1w1 = val_epoch(val_x,val_y)
            test_precision = test_1w1 / (test_1w1 + test_1w0) # TP/(TP+FP)
            test_recall = test_1w1 / (test_1w1 + test_0w1) # TP/(TP+FN)
            
            if save_path is not None:
                np.savez(save_path, *lasagne.layers.get_all_param_values(model))
        
        epoch_duration = time.time() - start_time
        
        # Then we print the results for this epoch:
        print("  \n ")
        print("Epoch "+str(epoch + 1)+" of "+str(num_epochs)+" took "+str(epoch_duration)+"s")
        print("  LR:                            "+str(LR))
        print("  training loss:                 "+str(train_loss))
        print("  validation loss:               "+str(val_loss))
        print("  validation error rate:         "+str(val_err)+"%")
        print("  best epoch:                    "+str(best_epoch))
        print("  best validation error rate:    "+str(val_err)+"%")
        print("  \n ")
        print("  test loss:                     "+str(test_loss))
        print("  test error rate:               "+str(test_err)+"%") 
        print("  False Negative:                "+str(test_0w1))
        print("  False Positive:                "+str(test_1w0))
        print("  True Positive:                 "+str(test_1w1))
        print("  Precision:                     "+str(test_precision)+"%")
        print("  Recall:                        "+str(test_recall)+"%")
        print("  \n ")
        print("  \n ")

def forward_pass(forward_pass_fn,X_test):
    
    label = forward_pass_fn(X)

    return label
