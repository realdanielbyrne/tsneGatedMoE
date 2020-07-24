# Notes

# vae testing
- 20 epochs seemed to be a sweet spot
- The VAE achieved lower losses with msle
- VD appeared to decrease loss in vae, but results were skewed.  VD appearded to reward circles and perpendicular lines.
  - 0, 9 and 6 were distinuishable
  - 1 was opposite 0, and 7 was rotated from both, but elongated
  - By contrast regular Sampling Layer has troubles discerning loops
  - msle improved loss and prod
  - Loss
    - loss greater with msle
    - loss greater with vd, but reconstruction is garbled
  - A trained vae improved the stack model in just a single episode of training. 


## VD

- training seems to happen all in the bias term, and a litle in log_sigma2 term
- theta term stays fairly stationary, indicating that the gradients do not seem to be adjusting this parameter

## mlp-stack 

- parameters of prior network seemed to influence convergence of subsequent network
- Features not fed into Dense
  - 48% on cifar 10 300,300,100
  - 49% with overfitting on 300,300,300
  - Turning off encoder training didn't seem to affect the outcome
- stack as simple as encoder over a single Dense laye is sufficient for 98% accuracy
- cifar10 gets stuck at 30%

## SD


MNIST
With softmax
  - poor performance, even on MNIST with encoder locked, 
  - poor performance with encoder unlocked as well

Without softmax 
  - encoder locked achievd 97.9% afer 20 epochs on mnist
  - decent performance after 20 epochs, encoder unlocked on mnist


CIFAR10
Merely chance (10%)


## PD
### Interleaved

MNIST
- decent, 97% in 20 epochs
- interleaved archtecture, with a PD layer after every dense
- low initial accuracy
- lowish final accuracy after 20 epochs
- training got stuck at same value

CIFAR10
- Merely chance (10%), encoder not trainable
- Noisey (19%), encoder trainable


## classpd

Creates a dropout filter for each class by taking the softmax of the x_test in a MOE structure

MNIST
- Training gets stuck at 97% even with regularization.

CIFAR10
- Tried multiple architectures
- best performers for 51%

## pdf

initializes gates with parameters derived from encoder
- The initialization parameters (mean and var from the prior) detroy the subsequent network's ability to learn Especially on cifar10.


## VDMOE

MNIST works, but underfits. Improvement stalls at low 90%s
CIFAR10 doesn't work at all.


## gatedmoe

MNIST
works, need to see if branches specialize

CIFAR10
Struggles to achieve over 25%

initialization strategy seems to hurt not help

## cgd_model