from __future__ import absolute_import, division, print_function, unicode_literals

__doc__= """ Example usage of parametric_tSNE.
Generate some simple data in high (14) dimension, train a model,
and run additional generated data through the trained model"""

import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import tensorflow as tf 
tf.keras.backend.clear_session()  # For easy reset of notebook state.
print(tf.__version__)

from tensorflow import keras
from tensorflow.keras import Model
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import plot_model,to_categorical
from tensorflow.keras.layers import Dense, Input, Dropout, Embedding
from tensorflow.keras.layers import Add, Subtract, Multiply, Concatenate, Reshape, Flatten, Permute
from tensorflow.keras.layers import add, subtract, multiply, concatenate, SpatialDropout1D, average, Lambda
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as K
import multiprocessing as mp
import utils
import argparse

# Setings
plt.style.use('ggplot')
test_data_tag = 'none'
batch_size = 100
low_dim = 2
nb_epoch = 50
shuffle_interval = nb_epoch + 1
n_jobs = 1
perplexity = 30.0 
override = True
dropout_rate = .4
concatfunc = add
summary = False

# Paths
ptne_model_path = 'models/ptsne_mp_cifar10.h5'
p_path = 'models/p'


def Hbeta(D, beta):
  P = np.exp(-D * beta)
  sumP = np.sum(P) + 1e-14
  H = np.log(sumP) + beta * np.sum(D * P) / sumP
  P = P / sumP
  return H, P

def KLdivergence(P, Y):
    alpha = K.int_shape(Y)[1] - 1.
    sum_Y = K.sum(K.square(Y), axis=1)
    eps = K.constant(10e-15)
    D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
    Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
    i = tf.constant(1 - np.eye(K.int_shape(Y)[0]),dtype=tf.float32)
    Q = Q*i
    Q = Q/K.sum(Q)
    Q = K.maximum(Q, eps)
    C = K.log(P) -  K.log(Q)
    C = K.sum(P * C)
    return C


def compute_joint_probabilities(samples, batch_size=batch_size, d=2, perplexity=30, tol=1e-5, verbose=0):
  v = d - 1

  # Initialize some variables
  n = samples.shape[0]
  batch_size = min(batch_size, n)

  # Precompute joint probabilities for all batches
  if verbose > 0: print('Precomputing P-values...')
  batch_count = int(n / batch_size)
  P = np.zeros((batch_count, batch_size, batch_size))
  for i, start in enumerate(range(0, n - batch_size + 1, batch_size)):
      curX = samples[start:start+batch_size]                   # select batch
      P[i], beta = x2p(curX, perplexity, tol, verbose=verbose) # compute affinities using fixed perplexity
      P[i][np.isnan(P[i])] = 0                                 # make sure we don't have NaN's
      P[i] = (P[i] + P[i].T) # / 2                             # make symmetric
      P[i] = P[i] / P[i].sum()                                 # obtain estimation of joint probabilities
      P[i] = np.maximum(P[i], np.finfo(P[i].dtype).eps)

  return P


def x2p(X, u=15, tol=1e-4, print_iter=500, max_tries=50, verbose=0):
  # Initialize some variables
  n = X.shape[0]                     # number of instances
  P = np.zeros((n, n))               # empty probability matrix
  beta = np.ones(n)                  # empty precision vector
  logU = np.log(u)                   # log of perplexity (= entropy)

  # Compute pairwise distances
  if verbose > 0: print('Computing pairwise distances...')
  sum_X = np.sum(np.square(X), axis=1)
  # note: translating sum_X' from matlab to numpy means using reshape to add a dimension
  D = sum_X + sum_X[:,None] + -2 * X.dot(X.T)

  # Run over all datapoints
  if verbose > 0: print('Computing P-values...')
  for i in range(n):

      if verbose > 1 and print_iter and i % print_iter == 0:
          print('Computed P-values {} of {} datapoints...'.format(i, n))

      # Set minimum and maximum values for precision
      betamin = float('-inf')
      betamax = float('+inf')

      # Compute the Gaussian kernel and entropy for the current precision
      indices = np.concatenate((np.arange(0, i), np.arange(i + 1, n)))
      Di = D[i, indices]
      H, thisP = Hbeta(Di, beta[i])

      # Evaluate whether the perplexity is within tolerance
      Hdiff = H - logU
      tries = 0
      while abs(Hdiff) > tol and tries < max_tries:

          # If not, increase or decrease precision
          if Hdiff > 0:
              betamin = beta[i]
              if np.isinf(betamax):
                  beta[i] *= 2
              else:
                  beta[i] = (beta[i] + betamax) / 2
          else:
              betamax = beta[i]
              if np.isinf(betamin):
                  beta[i] /= 2
              else:
                  beta[i] = (beta[i] + betamin) / 2

          # Recompute the values
          H, thisP = Hbeta(Di, beta[i])
          Hdiff = H - logU
          tries += 1

      # Set the final row of P
      P[i, indices] = thisP

  if verbose > 0:
      print('Mean value of sigma: {}'.format(np.mean(np.sqrt(1 / beta))))
      print('Minimum value of sigma: {}'.format(np.min(np.sqrt(1 / beta))))
      print('Maximum value of sigma: {}'.format(np.max(np.sqrt(1 / beta))))

  return P, beta



def create_ptsne_embedding_model(x_train):
  input = Input((x_train.shape[1],))
  x = Dense(512, activation="relu")(input)
  x = Dense(512, activation="relu")(x)
  x = Dense(2048, activation="relu")(x)
  x = Dropout(rate=.2, name = "ptsne_embedding")(x)
  
  ptsne = Dense(2, name="ptsne")(x)    
  model = Model(input, ptsne)
  
  if summary:
    model.summary()

  return model

def create_p(x_train):
    if not os.path.isfile('p.npy'):
        print("Calculating p_train.")
        P = compute_joint_probabilities(x_train,verbose = 0)  
        p_train = P.reshape(x_train.shape[0], -1)
        np.save('p',p_train)
    else:
        print('Loading P')
        p_train = np.load('p.npy')
        
    return p_train 

def fit_embedding_model(x_train):
  print('Creating Embedding model.')
  embedding_model = create_ptsne_embedding_model(x_train)

  embedding_model.compile(optimizer='adam', 
                          loss=KLdivergence, 
                          metrics=['accuracy'])
  
  embedding_model.fit (
    x_train, p_train,
    epochs=10, batch_size = batch_size, 
    shuffle = False, 
    verbose = 1)
  
  return embedding_model

def plot_ptsne_model(pred,y_test = None):
  plt.clf()
  fig = plt.figure(figsize=(5, 5))
  plt.scatter(pred[:, 0], pred[:, 1], c=y_test, marker='o', s=4, edgecolor='', alpha = 0.5)
  fig.tight_layout()


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-c", "--categorical",
                      default=False,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")
  parser.add_argument("-s", "--embedding_type",
                      default='mean',
                      help="embedding_type - sample: Samples a single x_test latent variable for each class\n\
                            mean: Averages all x_test latent variables")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


  # Create a callback that saves the model's weights
  checkpoint_path = "training\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=0)

  # load data
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]
  
  p_train = create_p(x_train)
  model = fit_embedding_model(x_train)

  pred = model.predict([x_test,x_test])
  c = np.argmax(y_test, axis=1)
  plot_ptsne_model(pred[0],c)


