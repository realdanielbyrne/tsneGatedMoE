from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

#tf keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import Model
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras import losses 
import utils


def load_data(args):
  if args.dataset == 'mnist':
    return  utils.load_minst_data(args.categorical)
  else:
    return  utils.load_cifar10_data(args.categorical)


class Encoding(layers.Layer):
  def __init__(self, num_outputs, **kwargs):
    super(Encoding, self).__init__(**kwargs)
    self.num_outputs = num_outputs

  def call(self, inputs):
    z_mean, z_var = inputs
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, self.num_outputs))
    z1, z2 = tf.zeros(batch,self.num_outputs)
    z1, z2 = z1 + .01
    z1 = z1 + z_mean
    z2 = z2 + z_var

    z_mean = tf.repeat(z_mean,repeats=self.num_outputs//dim, axis = -1)
    z_var = tf.repeat(z_var,repeats=self.num_outputs//dim, axis = -1)
    y = z_mean + tf.exp(0.5 * z_var) * epsilon
    return y

  def get_config(self):
    return {'num_outputs': self.num_outputs}

class ProbabilityDropout(layers.layer):
  def __init__(self, num_outputs, **kwargs):
    super(ProbabilityDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs

  def call(self, inputs):
    z_mean, z_var = inputs
    batch = K.shape(z_mean)[0]
    dim = K.shape(z_mean)[1]

    epsilon = K.random_normal(shape=(batch, self.num_outputs))
    z_mean = tf.repeat(z_mean,repeats=self.num_outputs//dim, axis = -1)
    z_var = tf.repeat(z_var,repeats=self.num_outputs//dim, axis = -1)
    y = z_mean + tf.exp(0.5 * z_var) * epsilon
    return y

  def get_config(self):
    return {'num_outputs': self.num_outputs}


def create_encoder(input_dim, intermediate_dim, latent_dim):
  encoder_input = layers.Input(shape=(input_dim,), name='encoder_input')
  x = layers.Dense(intermediate_dim, activation='relu')(encoder_input)
  z_mean = layers.Dense(latent_dim, name='z_mean')(x)
  z_var = layers.Dense(latent_dim, name='z_var')(x)
  z = Encoding(num_outputs = latent_dim)([z_mean, z_var])

  encoder = Model(encoder_input, [z_mean,z_var,z], name='encoder')
  return encoder

def create_decoder(latent_dim, intermediate_dim, output_dim):
  latent_inputs = keras.layers.Input(shape=(latent_dim,), name='latent_inputs')
  x = keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
  decoder_outputs = keras.layers.Dense(output_dim, activation='sigmoid')(x)  
  decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
  return decoder

def create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim = None):
  if output_dim is None:
    output_dim = input_dim

  encoder = create_encoder(input_dim,intermediate_dim, latent_dim)
  decoder = create_decoder(latent_dim,intermediate_dim, output_dim)
  encoder_input = encoder.get_layer("encoder_input").input  
  z_mean, z_var, z = encoder(encoder_input)
  decoder_output = decoder(z)
  vae = keras.Model(encoder_input, decoder_output, name = 'vae')

  # vae loss
  reconstruction_loss = losses.mse(encoder_input, decoder_output)
  reconstruction_loss *= input_dim
  kl_loss = 1 + z_var - K.square(z_mean) - K.exp(z_var)
  kl_loss = K.sum(kl_loss, axis=-1)
  kl_loss *= -0.5
  vae_loss = K.mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)
  vae.summary()
  return vae, encoder

def create_vae_mlp(input_dim, latent_dim, num_labels, encoder):

  encoder_input = encoder.get_layer("encoder_input").input
  z_mean, z_var, z = encoder(encoder_input)
  x = Encoding(num_outputs = input_dim)([z_mean, z_var])
  x = layers.add([x, encoder_input])
  x = layers.Dense(intermediate_dim, activation='relu')(x)
  x = layers.Dense(intermediate_dim, activation='relu')(x)
  z = Encoding(num_outputs = intermediate_dim)([z_mean, z_var])
  x = layers.add([x, z] , name = 'add2')
  x = layers.Dense(intermediate_dim, activation='relu')(x)
  out = layers.Dense(num_labels, activation="softmax")(x)
  vae_mlp = Model(encoder_input, out, name="vae_mlp")
  vae_mlp.summary()
  return vae_mlp

if __name__ == '__main__':

  intermediate_dim = 512
  batch_size = 128
  latent_dim = 16
  vae_epochs = 3
  epochs = 20
  args = utils.parse_cmd()
  (x_train, y_train), (x_test, y_test), num_labels, y_test_cat  = load_data(args)
  input_dim = output_dim = x_train.shape[-1]

  vae, encoder = create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim)
  vae.compile(optimizer='adam')

  # train the autoencoder
  vae.fit(x_train,
      epochs = vae_epochs,
      batch_size = batch_size,
      validation_data = (x_test, None))

  encoder.trainable = False
  encoder.compile()

  vae_mlp = create_vae_mlp(input_dim, latent_dim, num_labels, encoder)
  vae_mlp.compile(loss='categorical_crossentropy',
              optimizer="adam",
              metrics=['acc'])

  vae_mlp.fit(x_train, y_train,
            batch_size=batch_size,
            epochs=epochs)
  # utils.plot_encoding(encoder,
  #               [x_test, y_test],
  #               batch_size=batch_size,
  #               model_name="vae_mlp")

  # score trained model
  scores = vae_mlp.evaluate(x_test,
                          y_test_cat,
                          batch_size=batch_size,
                          verbose=0)
  print('Test loss:', scores[0])
  print('Test accuracy:', scores[1])

