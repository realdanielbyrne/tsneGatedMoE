from __future__ import absolute_import, division, print_function, unicode_literals

__doc__= """ Example usage of parametric_tSNE.
Generate some simple data in high (14) dimension, train a model,
and run additional generated data through the trained model"""

import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import plaidml.keras # used plaidml so I can run on any machine's video card regardless if it is NVIDIA, AMD or Intel.
import TSNEsp

# Using Tensorflow
#from tensorflow import keras
#from tensorflow.keras import Model
#from tensorflow.keras.datasets import cifar10
#from tensorflow.keras.utils import plot_model,to_categorical
#from tensorflow.keras.layers import Dense, Input, Dropout, Embedding
#from tensorflow.keras.layers import Add, Subtract, Multiply, Concatenate, Reshape, Flatten, Permute
#from tensorflow.keras.layers import add, subtract, multiply, concatenate, SpatialDropout1D, average, Lambda
#from tensorflow.keras.callbacks import Callback
#from tensorflow.keras import backend as K

# Using Base Keras
import keras
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Lambda, multiply
import keras.backend as K
from keras.initializers import RandomNormal
from keras.optimizers import SGD
from keras.utils import plot_model, to_categorical
from keras.datasets import cifar10


# Setings
plt.style.use('ggplot')
test_data_tag = 'none'
batch_size = 100
low_dim = 2
nb_epoch = 10
shuffle_interval = nb_epoch + 1
n_jobs = 1
perplexity = 30.0
override = True
dropout_rate = .4

# Colab paths
# ptne_model_path = '/content/gdrive/My Drive/Colab Notebooks/models/ptsne_mp_cifar10.h5'
# combined_model_path = '/content/gdrive/My Drive/Colab Notebooks/models/combined.h5'
# control_model_path = '/content/gdrive/My Drive/Colab Notebooks/models/control.h5'
# p_path = '/content/gdrive/My Drive/Colab Notebooks/models/p'

# local paths
ptne_model_path = './models/ptsne_mp_cifar10.h5'
combined_model_path = './models/combined.h5'
control_model_path = './models/control.h5'
p_path = './models/p'

def load_cifar10_data():
  # load the CIFAR10 data
  K.set_image_data_format('channels_first')
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255

  # Convert class vectors to binary class matrices.
  y_train = to_categorical(y_train, 10)
  y_test = to_categorical(y_test, 10)

  return (x_train, y_train), (x_test, y_test)

def create_ptsne_embedding_model(x_train):
  input = Input((x_train.shape[1],))
  x = Dense(512, activation="relu")(input)
  x = Dense(512, activation="relu")(x)
  x = Dense(2048, activation="relu")(x)
  x = Dropout(rate=.2, name = "ptsne_embedding")(x)

  ptsne = Dense(2, name="ptsne")(x)
  model = Model(input, ptsne)
  model.summary()

  return model

def create_combined_model(x_train):

  ptsne_in = Input((x_train.shape[1],),name="ptsne_in")
  x = Dense(512, activation="relu")(ptsne_in)
  x = Dense(512, activation="relu")(x)
  x = Dense(2048, activation="relu")(x)
  x = Dropout(rate=.2)(x)

  ptsne_out = Dense(2, activation="relu", name="ptsne_out")(x)
  ptsne = Dense(x_train.shape[1],activation="relu", name="ptsne")(ptsne_out)

  mlp_input = Input((x_train.shape[1],))
  concat = multiply([ptsne, mlp_input], name="concat")
  m = Dense(512, activation='relu')(concat)
  m = Dense(512, activation='relu')(m)
  m = Dense(2048, activation="relu")(m)
  m = Dropout(rate = dropout_rate)(m)

  mlp = Dense(10, activation='softmax', name="mlp")(m)

  model = Model(inputs=[ptsne_in, mlp_input], outputs=[ptsne_out, mlp])
  #model.summary()

  return model

def create_mlp_model(x_train):
  mlp_input = Input((x_train.shape[1],))
  m = Dense(512, activation='relu')(mlp_input)
  m = Dense(512, activation='relu')(m)
  m = Dense(2048, activation="relu")(m)
  m = Dropout(rate = dropout_rate)(m)

  mlp = Dense(10, activation='softmax',name="mlp")(m)
  model = Model(inputs= mlp_input, outputs = mlp)
  #model.summary()
  return model

def fit_control_model(x_train, override):
  if override or not os.path.exists(control_model_path):
    print('Creating CONTROL model')
    control = create_mlp_model(x_train)
    control.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    control.fit(
        x_train, y_train,
        epochs=nb_epoch, batch_size = batch_size,
        shuffle = False,
        verbose = 1)
    print('Saving CONTROL model')
    control.save(control_model_path,overwrite = True)

  else:
    print('{time}: Loading CONTROL from {model_path}'.format(time=datetime.datetime.now(), model_path=control_model_path))
    control = keras.models.load_model(control_model_path)

  return control

def fit_test_model(x_train, override, concatfunc):
  if override or not os.path.exists(combined_model_path):
    print('Creating TEST model.')
    model = create_combined_model(x_train)
    model.compile(optimizer='adam', loss=[TSNEsp.KLdivergence,'categorical_crossentropy'], metrics=['accuracy'])
    model.fit(
        [x_train,x_train], [p_train,y_train],
        epochs=nb_epoch, batch_size = batch_size,
        shuffle = False,
        verbose = 1)
    print('Saving TEST model')
    model.save(combined_model_path,overwrite = True)

  else:
    print('{time}: Loading TEST from {model_path}'.format(time=datetime.datetime.now(), model_path=combined_model_path))
    cust_object = {TSNEsp.KLdivergence.__name__: TSNEsp.KLdivergence}
    model = keras.models.load_model(combined_model_path, custom_objects = cust_object)

  return model

def create_p(x_train):
  print("Calculating p_train.")
  if override or not os.path.exists(p_path):
    P = TSNEsp.compute_joint_probabilities(x_train,batch_size=batch_size,verbose = 0)
    p_train = P.reshape(x_train.shape[0], -1)
    np.save(p_path,p_train)
  else:
    print('{time}: Loading P from {model_path}'.format(time=datetime.datetime.now(), model_path=p_path))
    p_train = np.load(p_path)
  return p_train

# score and compare models
def score_models(p_train,x_test,y_test):
  controlscore = control.evaluate([x_test,x_test], [p_train[:10000],y_test],batch_size = batch_size, verbose=1)
  print('Control Test Loss:', controlscore[0])
  print('Control Test Accuracy:', controlscore[1])

  testscore = model.evaluate([x_test,x_test], [p_train[:10000],y_test],batch_size = batch_size, verbose=1)
  print('Test Loss:', testscore[0])
  print('Test Accuracy:', testscore[1])

  return (controlscore, testscore)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):
    """Implements reparameterization trick by sampling
    from a gaussian with zero mean and std=1.

    Arguments:
        args (tensor): mean and log of variance of Q(z|X)

    Returns:
        sampled latent vector (tensor)
    """

    z_mean, z_log_var, output_dim = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, output_dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_ptsne_embedding_reparameterization_model(x_train):
  input = Input((x_train.shape[1],))
  x = Dense(512, activation="relu")(input)
  x = Dense(512, activation="relu")(x)
  x = Dense(2048, activation="relu")(x)
  x = Dropout(rate=.2, name = "ptsne_embedding")(x)
  ptsne = Dense(2, name="ptsne")(x)
  ptsne_var = Dense(2, name='ptsne_var')(x)

  model = Model(input, [ptsne,ptsne_var] )
  model.summary()
  return model

def create_combined_model2(x_train):
  input = Input((x_train.shape[1],))
  x = Dense(512, activation="relu")(input)
  x = Dense(512, activation="relu")(x)
  x = Dense(2048, activation="relu")(x)
  x = Dropout(rate=.2, name = "ptsne_embedding")(x)

  # Outputs
  ptsne = Dense(2, name="ptsne")(x)
  ptsne_var = Dense(2, name='ptsne_var')(x)

  # Instantiate embedding model
  embedding = Model(input, [ptsne,ptsne_var])
  embedding.summary()
  plot_model(embedding, to_file='ptsne_rp.png', show_shapes=True)

  # Build gated mixture of experts model
  z = Lambda(sampling, output_shape=(x_train.shapde[1],), name='z')([ptsne, ptsne_var,(x_train.shapde[1],)])
  concat = multiply([z, input], name="concat")
  m = Dense(512, activation='relu')(concat)
  m = Dense(512, activation='relu')(m)
  m = Dense(512, activation='relu')(m)
  m = Dense(2048, activation="relu")(m)
  m = Dropout(rate = dropout_rate)(m)
  mlp = Dense(10, activation='softmax',name="mlp")(m)

  model = Model(input, mlp)
  model.summary()
  return model

def fit_combined_model2(x_train, y_train):
  model = create_combined_model2(x_train)
  model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  model.fit(
    x_train, y_train,
    epochs=nb_epoch,
    batch_size = batch_size,
    shuffle = True,
    verbose = 1)
  return model

def fit_embedding_model(x_train, override):
  print('Creating Embedding model.')
  embedding_model = create_ptsne_embedding_model(x_train)
  embedding_model.compile(optimizer='adam', loss=TSNEsp.KLdivergence, metrics=['accuracy'])

  embedding_model.fit(
    x_train, p_train,
    epochs=10, batch_size = batch_size,
    shuffle = False,
    verbose = 0)

  return embedding_model

def plot_ptsne_model(pred,y_test = None):
  plt.clf()
  fig = plt.figure(figsize=(5, 5))
  plt.scatter(pred[:, 0], pred[:, 1], c=y_test, marker='o', s=4, edgecolor='', alpha = 0.5)
  fig.tight_layout()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  (x_train, y_train), (x_test, y_test) = load_cifar10_data()

  # calculate P
  p_train = create_p(x_train)

  # Create embedding model
  embedding_model = fit_embedding_model(x_train, override)

  # Fit Combined Model
  model2 = fit_combined_model2(x_train, y_train)

  model2score = model2.evaluate(x_test, y_test,batch_size = batch_size, verbose=1)
  print('Model2 Loss:', model2score[0])
  print('Model2 Accuracy:', model2score[1])

  # create models
  control = fit_control_model(x_train, override)
  model = fit_test_model(x_train, override, fit_test_model)

  # score models
  controlscore = control.evaluate(x_test, y_test,batch_size = batch_size, verbose=1)
  print('Control Test Loss:', controlscore[0])
  print('Control Test Accuracy:', controlscore[1])

  testscore = model.evaluate([x_test,x_test], [p_train[:10000],y_test],batch_size = batch_size, verbose=1)
  print('Test Loss:', testscore[0])
  print('Test Accuracy:', testscore[1])
  pred = model.predict([x_test,x_test])

  c = np.argmax(y_test, axis=1)
  plot_ptsne_model(pred[0],c)
