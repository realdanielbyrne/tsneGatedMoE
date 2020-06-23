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
  
  inputs = keras.layers.Input(shape = x_train.shape[-1], name='digits')
  y_in = keras.layers.Input(shape = y_train.shape[-1], name='y_in')
  x = ConstantGausianDropout(x_train.shape[-1], initial_values)([inputs,y_in])

  x = Dense(300, activation='relu')(x)
  #x = VarDropout(300)(x)

  x = Dropout(.2)(x)
  x = Dense(100, activation='relu')(x)
  x = Dropout(.2)(x)

  #x = VarDropout(100)(x)
  model_out = Dense(num_labels, activation = 'softmax', name='model_out')(x)
  model = Model([inputs,y_in], model_out)
  model.summary()
  return model

def custom_train(model,x_train,y_train, yt):
  # Instantiate an optimizer.
  optimizer = keras.optimizers.Adam()
  # Instantiate a loss function.
  loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
 
  # prepare data  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, yt))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

  for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

        # Open a GradientTape to record the operations run
        # during the forward pass, which enables autodifferentiation.
        with tf.GradientTape() as tape:

            # Run the forward pass of the layer.
            # The operations that the layer applies
            # to its inputs are going to be recorded
            # on the GradientTape.
            logits = model([x_batch_train, y_batch_train], training=True)  # Logits for this minibatch

            # Compute the loss value for this minibatch.
            loss_value = loss_fn(y_batch_train, logits)

        # Use the gradient tape to automatically retrieve
        # the gradients of the trainable variables with respect to the loss.
        grads = tape.gradient(loss_value, model.trainable_weights)

        # Run one step of gradient descent by updating
        # the value of the variables to minimize the loss.
        optimizer.apply_gradients(zip(grads, model.trainable_weights))

        # Log every 200 batches.
        if step % 200 == 0:
            print(
                "Training loss (for one batch) at step %d: %.4f"
                % (step, float(loss_value))
            )
            print("Seen so far: %s samples" % ((step + 1) * BATCH_SIZE))

# Setings
DIMENSION = 784
EPOCHS = 10
intermediate_dim = 512
BATCH_SIZE = 64
latent_dim = 2
vae_epochs = 1

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-s", "--sparse",
                      default=True,
                      help="Use sparse, integer encoding, instead of one-hot")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  model_name = 'basicmlp.h5'
  args = parser.parse_args()

  (x_train, y_train), (x_test, y_test),num_labels = utils.load_minst_data(categorical=False)
  input_dim = output_dim = x_train.shape[-1]
  
  vae, encoder = create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim)
  vae.compile(optimizer='adam')

  # train the autoencoder
  vae.fit(x_train,
      epochs = vae_epochs,
      batch_size = BATCH_SIZE,
      validation_data = (x_test, None))

  # fixed filter model
  z, _, _ = encoder.predict(x_test, batch_size=BATCH_SIZE) 
  
  initial_thetas = []
  initial_log_sigma2s = []

  for x in range(10):
    targets = np.where(y_test == x)[0]
    sample = targets[np.random.randint(targets.shape[0])]
    initial_thetas.append(z[sample][0])
    initial_log_sigma2s.append(z[sample][1])

  yt = y_train.reshape(y_train.shape[0],1)
  model = create_model(x_train, yt, [initial_thetas,initial_log_sigma2s], num_labels, encoder)

  # Train
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  custom_train(model, x_train, y_train, yt)

  #model.compile('adam',loss = 'sparse_categorical_crossentropy',metrics=['accuracy'])
  #model.fit([x_train, y_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_callback])

  # model accuracy on test dataset
  score = model.evaluate([x_test,y_test], y_test, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))
