from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse

# Using Base Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from layers.vd import VarDropout, ConstantGausianDropout
import utils




def dkl_qp(log_alpha):
    k1, k2, k3 = 0.63576, 1.8732, 1.48695; C = -k1
    mdkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.log1p(tf.exp(-log_alpha)) + C
    return -tf.reduce_sum(mdkl)

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

class Sampling(Layer):

  def call(self, inputs):
    z_mean, z_log_var = inputs
    batch = tf.shape(z_mean)[0]
    dim = tf.shape(z_mean)[1]
    epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_encoder(input_dim, intermediate_dim, latent_dim):
  encoder_input = Input(shape=(input_dim,), name='encoder_input')
  x = Dense(intermediate_dim, activation='relu')(encoder_input)
  z_mean = Dense(latent_dim, name='z_mean')(x)
  z_var = Dense(latent_dim, name='z_var')(x)
  z = Sampling()([z_mean, z_var])

  encoder = Model(encoder_input, [z_mean,z_var,z], name='encoder')
  return encoder

def create_decoder(latent_dim, intermediate_dim, output_dim):
  latent_inputs = keras.layers.Input(shape=(latent_dim,), name='latent_inputs')
  x = keras.layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
  decoder_outputs = keras.layers.Dense(output_dim)(x)  
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
  #vae.summary()
  return vae, encoder

def create_model(x_train, y_train, initial_values, num_labels, encoder):
  inputs = keras.layers.Input(shape = x_train.shape[-1], name='y_in')
  y_in = keras.layers.Input(shape = y_train.shape[-1], name='y_in')
  x = ConstantGausianDropout(x_train.shape[-1])([inputs,y_in])

  x = Dense(300, activation='relu')(x)
  x = VarDropout(300)(x)
  x = Dense(100, activation='relu')(x)
  x = VarDropout(100)(x)
  model_out = Dense(num_labels, activation='softmax', name='model_out')(x)
  model = Model([inputs,y_in], model_out)
  model.summary()
  return model


if __name__ == '__main__':

  # Setings
  DIMENSION = 784
  EPOCHS = 100
  BATCH_SIZE = 128
  intermediate_dim = 512
  batch_size = 128

  latent_dim = 2
  vae_epochs = 10

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-s", "--sparse",
                      default=True,
                      help="Use sparse, integer encoding, instead of one-hot")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'basicmlp.h5'
  args = parser.parse_args()


  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(True)
  input_dim = output_dim = x_train.shape[-1]
  
  vae, encoder = create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim)
  vae.compile(optimizer='adam')

  
  # train the autoencoder
  vae.fit(x_train,
      epochs = vae_epochs,
      batch_size = batch_size,
      validation_data = (x_test, None))

  # fixed filter model
  z, _, _ = encoder.predict(x_test, batch_size=batch_size)
  nb_classes = 10
  
  
  initial_values = [
    z[y_test ==0][0],
    z[y_test ==1][0],
    z[y_test ==2][0],
    z[y_test ==3][0],
    z[y_test ==4][0],
    z[y_test ==5][0],
    z[y_test ==6][0],
    z[y_test ==7][0],
    z[y_test ==8][0],
    z[y_test ==9][0],
  ]



  model = create_model(x_train, y_train, initial_values, num_labels, encoder)

  # Train
  model_input = model.get_layer("encoder_input").input  
  model_output = model.get_layer("model_out").output  
  model.compile('adam',loss = 'categorical_crossentropy',metrics=['accuracy'])

  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_callback])

  # model accuracy on test dataset
  score = model.evaluate(x_test, y_test_cat, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))
