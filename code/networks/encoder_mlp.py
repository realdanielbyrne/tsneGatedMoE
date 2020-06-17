from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model, to_categorical
from tensorflow.keras.datasets import mnist, cifar10


# Setings
plt.style.use('ggplot')
test_data_tag = 'none'
batch_size = 128
low_dim = 2
nb_epoch = 25
shuffle_interval = nb_epoch + 1
n_jobs = 1
perplexity = 30.0
override = True
dropout_rate = .45
intermediate_dim = 512
bottleneck_dim = 32
latent_dim = 16

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

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255

  if not sparse:
    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

  return (x_train, y_train), (x_test, y_test), num_labels

class Sampling(Layer):
  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon

def create_model(x_train, num_labels):
  # Encoder model
  original_inputs = Input(shape=(x_train.shape[1],), name='encoder_input')
  x = Dense(intermediate_dim, activation='relu')(original_inputs)
  z_mean = Dense(latent_dim, name='z_mean')(x)
  z_log_var = Dense(latent_dim, name='z_log_var')(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = Model(original_inputs, z, name='encoder')
  encoder.summary()

  # Decoder model
  latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
  di = Dense(intermediate_dim, activation='relu', name='decode_intermediate')(latent_inputs)
  decoder_outputs = Dense(x_train.shape[1], activation='sigmoid')(di)
  decoder = Model(latent_inputs, decoder_outputs, name='decoder')
  decoder.summary()

  # VAE model
  vae_outputs = decoder(encoder(original_inputs)[2])
  vae = Model(original_inputs, vae_outputs, name='vae_mlp')

  reconstruction_loss = mse(original_inputs, vae_outputs)
  reconstruction_loss *= x_train.shape[1]
  kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(kl_loss)
  vae.add_loss(vae_loss)

  # MLP Model with Encoded Inputs
  print(x_train.shape[1])
  mlp_inputs = Input(shape=(x_train.shape[1],), name='mlp_input')
  x = Dense(latent_dim, activation='relu')(mlp_inputs)
  x = BatchNormalization()(x)
  x = multiply([x,di])
  x = Dense(intermediate_dim, activation='relu')(x)
  #x = Dropout(dropout_rate)(x)
  x = multiply([x,di])
  x = Dense(intermediate_dim, activation='relu')(x)
  #x = Dropout(dropout_rate)(x)
  x = multiply([x,di])
  mlp_outputs = Dense(num_labels, activation='softmax')(x)
  model = Model([original_inputs, mlp_inputs], [vae_outputs,mlp_outputs])
  model.summary()
  return model


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='AE Embedding Classifier')


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

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'encoder_gating.h5'

  model = create_model(x_train,num_labels)

  # Train
  if args.sparse:
      loss_ = "sparse_categorical_crossentropy"
  else:
    loss_="categorical_crossentropy"

  model.compile('adam', loss=loss_, metrics=['accuracy'])
  model.fit([x_train, x_train], [x_train,y_train], epochs=nb_epoch, batch_size=batch_size)


  # model accuracy on test dataset
  score = model.evaluate([x_test,x_test], y_test, batch_size=batch_size)
  print('\nEncoder Test Loss:', score[0])
  print("Encoder Test Accuracy: %.1f%%" % (100.0 * score[1]))


