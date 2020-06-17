from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse
import matplotlib as plt

# Using Base Keras
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, Flatten,BatchNormalization,Activation
from tensorflow.keras import backend as K
from tensorflow.keras.initializers import RandomNormal
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.datasets import cifar10,mnist

# Setings

batch_size = 128
nb_epoch = 20
dropout_rate = .45

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
  parser = argparse.ArgumentParser(description='CNN Control Classifier')


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
  print('CNN Control Model Test Loss:', score[0])
  print("CNN Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))


