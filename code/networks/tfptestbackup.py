from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse

# Using Base Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.python.eager import context
import utils


# Settings
DIMENSION = 784
EPOCHS = 10
intermediate_dim = 512
BATCH_SIZE = 100
latent_dim = 2
vae_epochs = 1

import tensorflow_probability as tfp



def custom_train(model, x_train, y_train):


  
  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  # Instantiate a loss
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True)

  # Instantiate an optimizer.
  optimizer = keras.optimizers.Adam()
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
  val_accuracy = tf.keras.metrics.CategoricalAccuracy()

  # prepare data  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

  # Prepare the validation dataset.
  # Reserve 5,000 samples for validation.
  x_val = x_train[-5000:]
  y_val = y_train[-5000:]
  x_train = x_train[:-5000]
  y_train = y_train[:-5000]
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(BATCH_SIZE)

  @tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      logits = model(x_batch_train, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

      # Add kld layer losses created during this forward pass:
      kld_losses =  sum(model.losses) 
      kld_losses = kld_losses / float(x_train.shape[0])
      loss_value += kld_losses

      # Retrievethe gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(loss_value, model.trainable_weights)
      # Minimize the loss.
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metrics
      epoch_loss_avg.update_state(loss_value) 
      epoch_accuracy.update_state(y_batch_train,logits)
    return loss_value

  @tf.function
  def test_step(x, y):
      val_logits = model(x, training=False)
      val_accuracy.update_state(y, val_logits)

  for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

      # Run the forward pass.
      loss_value = train_step(x_batch_train, y_batch_train)

      # Log every 200 batches.
      if step % 200 == 0:
        print(
          "Training loss (for one batch) at step %d: %.4f"
            % (step, float(loss_value))
        )
      
    # Display metrics at the end of each epoch.
    print("Training acc over epoch: %.4f" % (float(epoch_accuracy.result()),))
    print("Training loss over epoch: %.4f" % (float(epoch_loss_avg.result()),))

    # Reset training metrics at the end of each epoch
    epoch_accuracy.reset_states()      
    epoch_loss_avg.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)

    val_acc = val_accuracy.result()
    val_accuracy.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))
    

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
  (x_train, y_train), (x_test, y_test),num_labels = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]

  model = tf.keras.Sequential([
      tfp.layers.DenseLocalReparameterization(300, activation=tf.nn.relu),
      tfp.layers.DenseLocalReparameterization(100, activation=tf.nn.relu),
      tfp.layers.DenseLocalReparameterization(100, activation=tf.nn.relu),
      tf.keras.layers.Dense(10),
  ])

  # use graph training for speed

  
  # use custom training loop to assist in debugging
  custom_train(model, x_train, y_train)

  metric_fn = tf.keras.metrics.CategoricalAccuracy
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True)



