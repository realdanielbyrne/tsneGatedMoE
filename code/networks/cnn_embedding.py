from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import plaidml.keras # used plaidml so I can run on any machine's video card regardless if it is NVIDIA, AMD or Intel.

# Using Base Keras
import keras
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Dropout, Conv2D, Flatten,BatchNormalization,Activation
import keras.backend as K
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.utils import plot_model, to_categorical
from keras.datasets import cifar10,mnist

# Setings
plt.style.use('ggplot')
test_data_tag = 'none'
batch_size = 128
low_dim = 2
nb_epoch = 20
shuffle_interval = nb_epoch + 1
n_jobs = 1
perplexity = 30.0
override = True
dropout_rate = .45
intermediate_dim = 512
latent_dim = 256

# local paths
ptne_model_path = 'models/ptsne_mp_cifar10.h5'
combined_model_path = 'models/combined.h5'
control_model_path = 'models/control.h5'
p_path = 'models/p.npy'

"""
# Standard model + embedding on front
Encoder Test Loss: 1.450488393306732
Encoder Test Accuracy: 47.7%

# Standard model + embedding on front + no Dropout
Encoder Test Loss: 1.4653235626220704
Encoder Test Accuracy: 51.6%

# Standard model + embedding instead of Dropout
Encoder Test Loss: 1.4653235626220704
Encoder Test Accuracy: 52.8%
"""
def load_minst_data(sparse):
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

  # convert to one-hot vector
  if not sparse:
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

  return (x_train, y_train), (x_test, y_test), num_labels

def load_cifar10_data(sparse):
  # load the CIFAR10 data
  K.set_image_data_format('channels_first')
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  # compute the number of labels
  num_labels = len(np.unique(y_train))

  x_train = x_train.astype('float32') / 255
  x_test = x_test.astype('float32') / 255

  if not sparse:
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

  return (x_train, y_train), (x_test, y_test), num_labels

def create_model(x_train, num_labels):
  model = Sequential()
  model.add(Conv2D(input_shape=x_train[0,:,:,:].shape, filters=96, kernel_size=(3,3)))
  model.add(Activation('relu'))
  model.add(Conv2D(filters=96, kernel_size=(3,3), strides=2))
  model.add(Activation('relu'))
  model.add(Dropout(0.2))
  model.add(Conv2D(filters=192, kernel_size=(3,3)))
  model.add(Activation('relu'))
  model.add(Conv2D(filters=192, kernel_size=(3,3), strides=2))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Flatten())
  model.add(BatchNormalization())
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dense(num_labels, activation="softmax"))

  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='CNN Embedding Classifier')


  parser.add_argument("-s", "--sparse",
                      action='store_false',
                      help="Use sparse, integer encoding, instead of one-hot")

  parser.add_argument("-ds", "--dataset",
                      action='store',
                      type=str,
                      default='cifar10',
                      help="Use sparse, integer encoding, instead of one-hot")

  args = parser.parse_args()
  if args.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test), num_labels = load_minst_data(args.sparse)
  else:
    (x_train, y_train), (x_test, y_test), num_labels = load_cifar10_data(args.sparse)


  model = create_model(x_train,num_labels)

  # Train
  if args.sparse:
    loss_ = "sparse_categorical_crossentropy"
  else:
    loss_="categorical_crossentropy"


  callbacks_list = None
  model.compile(loss=loss_, optimizer='adam', metrics=['accuracy'])

  H = model.fit(x_train, y_train, validation_data=(x_test, y_test),
              epochs=nb_epoch, batch_size=batch_size, callbacks=callbacks_list)

  # model accuracy on test dataset
  score = model.evaluate(x_test, y_test, batch_size=batch_size)
  print('\CNN Control Model Test Loss:', score[0])
  print("CNN Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))


