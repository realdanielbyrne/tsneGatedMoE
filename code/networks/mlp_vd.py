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
from layers.vd import VarDropout
import utils

# Setings
DIMENSION = 784
EPOCHS = 20
BATCH_SIZE = 128
intermediate_dim = 512
batch_size = 128
latent_dim = 16
vae_epochs = 4

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

class  ProbabilityDropout(Layer):
  def __init__( self, 
                zero_point = 1e-10,
                **kwargs):
    super(ProbabilityDropout, self).__init__(**kwargs)
    self.zero_point = zero_point

  def build(self, input_shape):
    z_mean_shape, _, in_shape = input_shape
    self.dim = z_mean_shape[-1]
    self.num_outputs = in_shape[-1] 
    return super().build(input_shape)

  def call(self, inputs, training = None):
    z_mean, z_var, x = inputs
    batch = K.shape(z_mean)[0]
    multiplier = self.num_outputs // batch

    if training :
      z_mean = tf.repeat(z_mean,repeats = multiplier, axis = 0)
      z_var = tf.repeat(z_var,repeats = multiplier, axis = 0)

      epsilon = K.random_normal(shape=(batch * multiplier, self.dim))
      z = z_mean + tf.exp(0.5 * z_var) * epsilon
      z = tf.reshape(z,shape=(batch,self.dim * multiplier))

      # computes drop probability
      def dropprob(p):
        p_range = [K.min(p,axis = -1),K.max(p, axis = -1)]
        p = tf.nn.softmax(tf.cast(tf.histogram_fixed_width(p, p_range, nbins = self.num_outputs),dtype=tf.float32))
        return p

      probs = tf.map_fn(dropprob, z)


      # push values that are close to zero, to zero, promotes sparse models which are more efficient
      #condition = tf.less(probs,self.zero_point)
      #probs = tf.where(condition,tf.zeros_like(probs),probs)
      
      # scales output after zers to encourage sum to be similar to sum before zeroing out connections
      scale_factor = tf.cast(1 / multiplier,tf.float32)
      return x * probs / scale_factor
    return x

  def get_config(self):
    return {"zero_point":zero_point}
      
  def compute_output_shape(self, input_shape):
    _, _, in_shape = input_shape
    return in_shape[-1]


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

def create_model(x_train, num_labels, encoder):
  encoder_input = encoder.get_layer("encoder_input").input
  z_mean, z_var, z = encoder(encoder_input)

  #model_in = Input(shape=(x_train.shape[1],), name='model_in')

  x = ProbabilityDropout( name="prob_dropout")([z_mean,z_var,encoder_input])
  x = Dense(300, activation='relu')(x)
  x = VarDropout(300)(x)
  x = Dense(100, activation='relu')(x)
  x = VarDropout(100)(x)
  model_out = Dense(num_labels, activation='softmax', name='model_out')(x)
  model = Model(encoder_input, model_out)
  model.summary()
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
  input_dim = output_dim = x_train.shape[-1]

  vae, encoder = create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim)
  vae.compile(optimizer='adam')

  # train the autoencoder
  vae.fit(x_train,
      epochs = vae_epochs,
      batch_size = batch_size,
      validation_data = (x_test, None))

  encoder.trainable = False
  #encoder.compile()

  model = create_model(x_train, num_labels, encoder)

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
