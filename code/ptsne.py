import numpy as np
np.random.seed(71)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation

from keras import backend as K
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.callbacks import Callback
from keras.utils import np_utils
from keras.objectives import categorical_crossentropy
from keras.datasets import cifar10, mnist
import multiprocessing as mp
import math
import tensorflow as tf
import datetime

def Hbeta(D, beta):
  P = np.exp(-D * beta)
  sumP = np.sum(P) + 10e-15
  H = np.log(sumP) + beta * np.sum(D * P) / sumP
  P = P / sumP
  return H, P

def x2p_job(data):
  i, Di, tol, logU = data
  beta = 1.0
  betamin = -np.inf
  betamax = np.inf
  H, thisP = Hbeta(Di, beta)

  Hdiff = H - logU
  tries = 0
  while np.abs(Hdiff) > tol and tries < 50:
      if Hdiff > 0:
          betamin = beta
          if betamax == -np.inf:
              beta = beta * 2
          else:
              beta = (betamin + betamax) / 2
      else:
          betamax = beta
          if betamin == -np.inf:
              beta = beta / 2
          else:
              beta = (betamin + betamax) / 2

      H, thisP = Hbeta(Di, beta)
      Hdiff = H - logU
      tries += 1

  return i, thisP

def x2p(X):
  tol = 1e-5
  n = X.shape[0]
  logU = np.log(perplexity)

  sum_X = np.sum(np.square(X), axis=1)
  D = sum_X + (sum_X.reshape([-1, 1]) - 2 * np.dot(X, X.T))

  idx = (1 - np.eye(n)).astype(bool)
  D = D[idx].reshape([n, -1])

  def generator():
      for i in range(n):
          yield i, D[i], tol, logU

  pool = mp.Pool(n_jobs)
  result = pool.map(x2p_job, generator())
  P = np.zeros([n, n])
  for i, thisP in result:
      P[i, idx[i]] = thisP

  return P

def calculate_P(X):
  print ("Computing pairwise distances...")
  n = X.shape[0]
  P = np.zeros([n, batch_size])
  for i in range(0, n, batch_size):
      P_batch = x2p(X[i:i + batch_size])
      P_batch[np.isnan(P_batch)] = 0
      P_batch = P_batch + P_batch.T
      P_batch = P_batch / P_batch.sum()
      P_batch = np.maximum(P_batch, 1e-12)
      P[i:i + batch_size] = P_batch
  return P

def KLdivergence(P, Y):
  low_dim = 2
  alpha = low_dim - 1.
  sum_Y = K.sum(K.square(Y), axis=1)
  eps = K.variable(10e-15)
  D = sum_Y + K.reshape(sum_Y, [-1, 1]) - 2 * K.dot(Y, K.transpose(Y))
  Q = K.pow(1 + D / alpha, -(alpha + 1) / 2)
  Q *= K.variable(1 - np.eye(batch_size))
  Q /= K.sum(Q)
  Q = K.maximum(Q, eps)
  C = K.log(P) - K.log(Q)
  C = K.sum(P * C)
  return C

def build_ptsne_model(dataset):
  (x_train,x_test,y_train,y_test) = dataset

  perplexity = 30.0
  n_jobs = 4
  batch_size = 100

  nb_epoch = 50
  shuffle_interval = nb_epoch + 1

  batch_num = int(n // batch_size)
  m = batch_num * batch_size

  # build model
  inputs = Input(shape=(x_train.shape[1]))

  x = Dense(500, activation='relu')(inputs)
  x = Dense(500, activation='relu')(x)
  x = Dropout(0.2)(x)
  predictions = Dense(10, activation='softmax')(x)

  # This creates a model that includes
  # the Input layer and three Dense layers
  model = Model(inputs=inputs, outputs=predictions)
  model.compile(optimizer='adam',
                loss=KLdivergence,
                metrics=['accuracy'])

  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  model.fit(x=x_train,
            y=y_train,
            epochs=5,
            validation_data=(x_test, y_test),
            callbacks=[tensorboard_callback])
  return model

