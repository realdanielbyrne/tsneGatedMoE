from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__doc__= """ Example usage of parametric_tSNE.
Generate some simple data in high (14) dimension, train a model,
and run additional generated data through the trained model"""

import sys,os, datetime, shutil, zipfile, glob,math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load Keras and PlaidML
import plaidml.keras
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout
from keras.callbacks import Callback
from keras import backend as K
from tsne import TSNE


# Globals
plt.style.use('ggplot')
num_classes = 10
cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)

num_classes = 10
test_data_tag = 'none'
batch_size = 5000
low_dim = 2
nb_epoch = 20
shuffle_interval = nb_epoch + 1
n_jobs = 8
perplexity = 30.0

color_palette = sns.color_palette("hls", num_classes)
model_path_template = 'ptsne_{model_tag}_{test_data_tag}.h5'
figure_template = 'ptsne_viz_tSNE_{test_data_tag}.pdf'
log_dir = '.\output'

def load_cifar10_data():
  # load the CIFAR10 data
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255

  # Convert class vectors to binary class matrices.
  #y_train_cat = to_categorical(y_train, num_classes)
  #y_test_cat = to_categorical(y_test, num_classes)

  return (x_train,x_test),(y_train,y_test)

def create_model(x_train):
  input = Input((x_train.shape[1],))
  x = Dense(500, activation="relu")(input)
  x = Dense(500, activation="relu")(x)
  x = Dense(1000, activation="relu")(x)
  x = Dropout(0.2)(x)
  x = Dense(2)(x)

  model = Model(input, x)
  return model

def generate_batch(x_train,n):
  for i in range(0, n, batch_size):
      P = calculate_P(x_train[i:i+batch_size])
      yield(x_train[i:i+batch_size], P[i:i+batch_size])

def generator(x_train):
  while True:
      indices = np.arange(x_train.shape[0])
      np.random.shuffle(indices)
      for i in range(x_train.shape[0]//batch_size):
          current_indices = indices[i*batch_size:(i+1)*batch_size]
          X_batch = x_train[current_indices]
          # X to P
          P = TSNE.calculate_P(X_batch)
          yield X_batch, P

def main():
  (x_train, y_train), (x_test, y_test) = load_cifar10_data()
  model = create_model(x_train)
  model.compile("adam", loss = TSNE.KLdivergence)
  #fit_model(model,n,x_train,y_train,x_test,y_test)
  #cb = Sampling(model, x_train[:1000], y_train[:1000])
  #model.fit_generator(generator(x_train,batch_size), steps_per_epoch=x_train.shape[0]//batch_size,epochs=nb_epoch, callbacks=[cb])
  history = model.fit_generator(
      generator(x_train),
      steps_per_epoch=math.ceil(x_train.shape[0]//batch_size),
      epochs=nb_epoch)

main()