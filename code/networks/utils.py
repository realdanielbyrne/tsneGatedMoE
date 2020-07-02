from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, mnist



#######################################################
# handy function to keep track of sparsity
def sparseness(log_alphas, thresh=3):
    N_active, N_total = 0., 0.
    for la in log_alphas:
        m = tf.cast(tf.less(la, thresh), tf.float32)
        n_active = tf.reduce_sum(m)
        n_total = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
        N_active += n_active
        N_total += n_total
    return 1.0 - N_active/N_total


def parse_cmd(description = 'AE Embedding Classifier'):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-c", "--categorical",
                      action='store_false',
                      help="Use one-hot encoding.")

  parser.add_argument("-ds", "--dataset",
                      action='store',
                      type=str,
                      default='mnist',
                      help="Use sparse, integer encoding, instead of one-hot")
                  


  args = parser.parse_args()
  return args

def load_minst_data(categorical):
      # load mnist dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
  # compute the number of labels
  num_labels = len(np.unique(y_train))

  # image dimensions (assumed square)
  image_size = x_train.shape[1]
  input_size = image_size * image_size

  # resize and normalize
  x_train = np.reshape(x_train, [-1, input_size]).astype('float32') / 255
  x_test = np.reshape(x_test, [-1, input_size]).astype('float32') / 255

  if categorical:
    # Convert class vectors to binary class matrices ( One Hot Encoding )
    y_train = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)

  return (x_train, y_train), (x_test, y_test), num_labels, y_test_cat

def load_cifar10_data(categorical):
  # load the CIFAR10 data
  K.set_image_data_format('channels_first')
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  # compute the number of labels
  num_labels = len(np.unique(y_train))

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255

  if categorical:
    # Convert class vectors to binary class matrices ( One Hot Encoding )
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)
    y_test_cat = to_categorical(y_test)

  return (x_train, y_train), (x_test, y_test), num_labels, y_test_cat



def plot_encoding(encoder,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()
