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
  x = Dense(intermediate_dim//2, activation='relu')(x)

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

def create_model(
                x_train, 
                yt,
                initial_values, 
                num_labels, 
                encoder, 
                model_name,
                dropout_type = 'var'):
  
  # Define Inputs
  inputs = keras.layers.Input(shape = (x_train.shape[-1],), name='digits')
  y_in = keras.layers.Input(shape = (yt.shape[-1],), name='y_in')

  # ConstantGausianDropout first to establish a pattern
  x = ConstantGausianDropout(x_train.shape[-1], initial_values)([inputs,y_in])
  
  U = 300
  x = Dense(U, activation='relu')(x)
  if dropout_type is 'var':
    x = VarDropout(U)(x)
  else: 
    x = Dropout(.2)(x)

  U = 100
  x = Dense(U, activation='relu')(x)
  if dropout_type is 'var':
    x = VarDropout(U)(x)
  else: 
    x = Dropout(.2)(x)

  U = 100
  x = Dense(U, activation='relu')(x)
  if dropout_type is 'var':
    x = VarDropout(U)(x)
  else: 
    x = Dropout(.2)(x)

  model_out = Dense(num_labels, activation = 'softmax', name='model_out')(x)
  model = Model([inputs,y_in], model_out, name = model_name)
  model.summary()
  
  return model

def custom_train(model,x_train,y_train, yt, loss_fn):
  # Instantiate an optimizer.
  optimizer = keras.optimizers.Adam()

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

def get_varparams_class_samples(predictions, y_test):
  initial_thetas = []
  initial_log_sigma2s = []
  
  for x in range(num_labels):
    targets = np.where(y_test == x)[0]
    sample = targets[np.random.randint(targets.shape[0])]
    initial_thetas.append(predictions[sample][0])
    initial_log_sigma2s.append(predictions[sample][1])
  
  return initial_thetas,initial_log_sigma2s

def get_varparams_class_means(predictions, y_test):
  initial_thetas = []
  initial_log_sigma2s = []

  for x in range(num_labels):
    targets = predictions[np.where(y_test == x)[0]]
    means = np.mean(targets, axis = 0)
    initial_thetas.append(means[0])
    initial_log_sigma2s.append(means[1])
    
  return initial_thetas,initial_log_sigma2s

# Settings
DIMENSION = 784
EPOCHS = 1
intermediate_dim = 512
BATCH_SIZE = 64
latent_dim = 2
vae_epochs = 1


if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-c", "--categorical",
                      default=False,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")
  parser.add_argument("-s", "--embedding_type",
                      default='mean',
                      help="embedding_type - sample: Samples a single x_test latent variable for each class\n\
                            mean: Averages all x_test latent variables")

  args = parser.parse_args()

  # parameters
  model_name = 'mlp_cgvd.h5'
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

  # Create a callback that saves the model's weights
  checkpoint_path = "training\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  # load data
  (x_train, y_train), (x_test, y_test),num_labels = utils.load_minst_data(categorical=False)
  input_dim = output_dim = x_train.shape[-1]
  
  # create and compile vae
  vae, encoder = create_vae_model(input_dim, latent_dim, intermediate_dim, output_dim)
  vae.compile(optimizer='adam')

  if os.path.isfile(checkpoint_path):
    vae.load_weights(checkpoint_path)
  
  # train the vae
  vae.fit(x_train,
      epochs = vae_epochs,
      batch_size = BATCH_SIZE,
      validation_data = (x_test, None),
      #callbacks=[cp_callback]
      )
  vae.save_weights(checkpoint_path)

  # gather predictions for the test batch
  predictions, _, _ = encoder.predict(x_test, batch_size=BATCH_SIZE) 

  if args.embedding_type == 'sample':
    initial_values = get_varparams_class_samples(predictions, y_test)
  else:
    initial_values = get_varparams_class_means(predictions, y_test)

  yt = y_train.reshape(y_train.shape[0],1)

  # create model under test
  model = create_model(x_train, yt, initial_values, num_labels, encoder, model_name)
  
  # Define First term of ELBO Loss 
  # kl loss is collected at each layer
  if args.categorical:
    loss_fn = keras.losses.CategoricalCrossentropy(from_logits = False)
  else:
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits = False)
  
  # Train
  # use custom training loop to assist in debugging
  custom_train(model, x_train, y_train, yt, loss_fn)
  
  # use graph training for speed
  #model.compile('adam',loss = loss_fn, metrics=['accuracy'])
  #model.fit([x_train, y_train], y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_callback])

  # model accuracy on test dataset
  score = model.evaluate([x_test,y_test], y_test, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))

