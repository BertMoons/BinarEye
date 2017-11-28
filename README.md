# BinarEye training code

This code is not fully cleaned-up yet, porting might not be trivial yet.  Expect regular updates to improve upon this.

## Introduction
Train your own __Binary Neural Networks (BNN)__ - networks trained with 1b weights and activations - in __Theano / Lasagne__.
This code was used to train Binary Neural Networks, specifically for the BinarEye chip (reference coming up).

This code uses / is based on the [lasagne/theano](https://github.com/MatthieuCourbariaux/BinaryNet) version of [BinaryNet](https://papers.nips.cc/paper/6573-binarized-neural-networks).

## Preliminaries
Running this code requires:
1. [Theano](http://deeplearning.net/software/theano/)
2. [Lasagne](https://lasagne.readthedocs.io/en/latest/)
4. A GPU with recent versions of [CUDA and CUDNN](https://developer.nvidia.com/cudnn)
3. [pylearn2](http://deeplearning.net/software/pylearn2/) for [MNIST](http://yann.lecun.com/exdb/mnist/), [SVHN](http://ufldl.stanford.edu/housenumbers/) and [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html)
4. The custom datasets for facedetection, ownerrecognition, anglerecognition and 10-face facerecognition can be found [here](http://homes.esat.kuleuven.be/~bmoons/)

## Training your own BNN

Once you have all data sets local, you can run the following commands to start training:

* cd ./CIFAR10/; python cifar10.py -f 256
* cd ./MNIST/; python mnist.py -f 256
* cd ./SVHN/; python svhn.py -f 256
* cd ./train_anglerecognition_custom/; python AngleSense.py -f 256
* cd ./train_facedetection_custom/; python AngleSense.py -f 256
* cd ./train_facerecognition_custom/; python AngleSense.py -f 256
* cd ./train_ownerrecognition_custom/; python AngleSense.py -f 256

Where f is the number of filters in every layer (the width of every layer in the network).
In BinarEye, f is constrained to either 64, 128 or 256.
 

## References and acknowledgements 

The used datasets are built on existing datasets. 

1. facedetection, ownerdetection and facerecognition are built on a combination of mini-crops of backgrounds from [Stanford](http://dags.stanford.edu/projects/scenedataset.html) and faces from [LFW](http://vis-www.cs.umass.edu/lfw/)
2. anglerecognition is built on cropped and aligned faces from [AdienceFaces](http://www.openu.ac.il/home/hassner/Adience/data.html)

All input images are rescaled to 32x32x3 input images in order to be processed by the BinarEye chip. Backgrounds are either 32x32 crops of larger images, or 32x32 rescales of larger sub-images.










