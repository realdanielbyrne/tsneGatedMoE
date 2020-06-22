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
DIMENSION = 300
EPOCHS = 20
BATCH_SIZE = 100

# local paths
ptne_model_path = 'models/ptsne_mp_cifar10.h5'
combined_model_path = 'models/combined.h5'
control_model_path = 'models/control.h5'
p_path = 'models/p.npy'

def create_model(x_train, num_labels):

  model_in = Input(shape=(x_train.shape[1],), name='model_in')
  x = Dense(300, activation='relu')(model_in)
  x = Dropout(DROPOUT_RATE)(x)
  x = Dense(100, activation='relu')(x)
  x = Dropout(DROPOUT_RATE)(x)
  model_out = Dense(num_labels, activation='softmax')(x)
  model = Model(model_in, model_out)

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
  (x_train, y_train), (x_test, y_test),num_labels = utils.load_minst_data(True)

  model = create_model(x_train,num_labels)

  # Train
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  model.compile('adam', loss="categorical_crossentropy",metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_callback])


  # model accuracy on test dataset
  score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))
