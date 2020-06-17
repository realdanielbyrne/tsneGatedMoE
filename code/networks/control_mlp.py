from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

# Using Base Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer

from tensorflow.keras.datasets import cifar10, mnist
import utils

# Setings
DROPOUT_RATE = .45
DIMENSION = 768
EPOCHS = 5
BATCH_SIZE = 100

# local paths
ptne_model_path = 'models/ptsne_mp_cifar10.h5'
combined_model_path = 'models/combined.h5'
control_model_path = 'models/control.h5'
p_path = 'models/p.npy'

def create_model(x_train, num_labels):
  model = Sequential()
  model.add(Dense(DIMENSION, input_dim=x_train.shape[1]))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT_RATE))

  model.add(Dense(DIMENSION))
  model.add(Activation('relu'))
  model.add(Dropout(DROPOUT_RATE))

  model.add(Dense(num_labels))
  model.add(Activation('softmax'))
  return model

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-s", "--sparse",
                      action='store_false',
                      help="Use sparse, integer encoding, instead of one-hot")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'basicmlp.h5'

  args = parser.parse_args()
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(args.sparse)

  model = create_model(x_train,num_labels)

  # Train
  model.compile('adam', loss="categorical_crossentropy",metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE)


  # model accuracy on test dataset
  score = model.evaluate(x_test, y_test_cat, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))
