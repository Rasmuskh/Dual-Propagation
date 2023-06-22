# Dual-Propagation
Code for the paper *Dual Propagation: Accelerating Contrastive Hebbian Learning with Dyadic Neurons*
## JAX_CNN_experimentswere
- This directory contains code necessary for running the CNN experiments on CIFAR10, CIFAR100 and ImageNet32x32.
- These experiments use the most effecient variant of dual propagation (described in the paper), which we have implemented on top of JAX autodiff for effeciency.

## julia_Flux_MLP_experiments
- This directory contains the necessary code for training an MLP on MNIST using various flavors of dual propagation (described in the paper). 
- We chose to implement these experiments in julia to take advantage of multiple dispatch and because some of the stranger flavors required multiple nested for loops, which julia deals better with than python.