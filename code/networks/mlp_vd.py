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
import tensorflow_model_optimization as tfmot
#from layers.vd import VarDropout, ConstantGausianDropout
import utils


#######################################################
# vd

EPSILON = 1e-8

class ConstantGausianDropoutGate(Layer):
  def __init__( self,
                initial_values,
                activation = tf.keras.activations.relu,
                use_bias = True,
                threshold=3.,
                clip_alpha=None,
                eps=EPSILON,
                value_limit = 8.,
                **kwargs):

    super(ConstantGausianDropoutGate, self).__init__(**kwargs)
    self.activation = activation
    self.use_bias = use_bias
    self.threshold = threshold
    self.clip_alpha = clip_alpha
    self.eps = eps
    self.value_limit = value_limit

    # unpack variational parameters, and extrapolate the number of classes for the lookup  
    initial_theta, initial_log_sigma2 = initial_values

    # define static lookups for pre-calculated datasets
    self.theta = initial_theta
    self.log_sigma2 = initial_log_sigma2

  def call(self, inputs, training = None):
    num_outputs = tf.shape(inputs)[-1]
    noise_shape = [tf.shape(inputs)[-1], num_outputs]
    theta = self.theta
    log_sigma2 = self.log_sigma2
    
    # repack parameters for convience 
    variational_params = (theta, log_sigma2)

    # compute dropout rate
    log_alpha = compute_log_alpha(variational_params) 

    # Compute log_sigma2 again so that we can clip on the log alpha magnitudes
    if self.clip_alpha is not None:
      log_sigma2 = compute_log_sigma2(log_alpha, theta)

    if training:
      mu = inputs * theta
      std = tf.sqrt(tf.square(inputs) * tf.exp(log_sigma2) + EPSILON)
      val = mu + tf.matmul(std, tf.random.normal(noise_shape))        

    else:
      log_alpha = compute_log_alpha(variational_params) 
      threshold = self.threshold
      weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)
      val = inputs * theta * weight_mask  

    # Apply an activation function to the output
    if self.activation is not None:
      val = self.activation(val)
     
    if not context.executing_eagerly():
      # Set the static shape for the result since it might lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      val.set_shape(self.compute_output_shape(inputs.shape))
    
    return val
      
  def compute_output_shape(self, input_shape):
    return  input_shape
  
  def get_config(self):
    config = {
      'activation':self.activation,
      'use_bias':self.use_bias,
      'threshold':self.threshold, 
      'clip_alpha':self.clip_alpha, 
      'eps':self.eps,
      'value_limit':self.value_limit 
    }
    base_config = super(VarDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

class VarDropout(Layer):
  def __init__(self,
               num_outputs = None,
               activation = tf.keras.activations.relu,
               kernel_initializer = tf.keras.initializers.RandomNormal,
               kernel_regularizer = tf.keras.regularizers.l2(.001),
               log_sigma2_initializer = None,
               use_bias=False,
               eps=EPSILON,
               threshold=3.,
               clip_alpha=8.,
               warmup = True,
               max_step = 10000,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.kernel_regularizer = kernel_regularizer
    self.log_sigma2_initializer = log_sigma2_initializer
    self.use_bias = use_bias
    self.eps = eps
    self.threshold = threshold
    self.clip_alpha = clip_alpha
    self.warmup = warmup
    self.step = tf.Variable(initial_value=1., trainable=False)
    self.max_step = max_step

  def build(self, input_shape):
    if self.num_outputs is None:
      self.num_outputs = input_shape[-1]

    kernel_shape = (input_shape[-1],self.num_outputs)

    self.theta = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            trainable=True)

    if self.log_sigma2_initializer is None:
      self.log_sigma2_initializer = tf.random_uniform_initializer()

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            trainable=True)

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.constant_initializer(0.),
                            trainable = True)
    else:
      self.b = None

    
  def call(self, inputs, training = None):

    if training:
      if self.clip_alpha is not None:
        # Compute the log_alphas and then compute the
        # log_sigma2 again so that we can clip on the
        # log alpha magnitudes
        log_alpha = compute_log_alpha(self.log_sigma2, self.theta)
        if self.clip_alpha is not None:
          # If a limit is specified, clip the alpha values
          log_alpha = tf.clip_by_value(log_alpha, -self.clip_alpha, self.clip_alpha)  
        log_sigma2 = compute_log_sigma2(self.theta, log_alpha)
      
      # Compute the mean and standard deviation of the distributions over the
      # activations
      mu = tf.matmul(inputs,self.theta)
      std = tf.sqrt(tf.matmul(tf.square(inputs),tf.exp(self.log_sigma2)) + self.eps)
      val = tf.random.normal(tf.shape(std),mu,std)
      
      # Constants
      k1, k2, k3, c = 0.63576, 1.8732, 1.48695, -0.63576

      # Compute element-wise dkl
      eltwise_dkl = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha) -0.5 * tf.math.log1p(tf.exp(-log_alpha)) + c
      dkl_loss = -tf.reduce_sum(eltwise_dkl) / 57000. # good for 60,000 training examples minu

      if self.warmup:
        self.step.assign_add(1.)
        fraction = tf.minimum(self.step / (self.max_step), 1.0)
        dkl_loss *= fraction

    else:
      #def compute_log_alpha(theta, log_sigma2):
      log_alpha = compute_log_alpha(self.theta, self.log_sigma2)
      weight_mask = tf.cast(tf.less(log_alpha, self.threshold), tf.float32)
      val = tf.matmul(inputs, self.theta * weight_mask)
      dkl_loss = 0.

    if self.use_bias:
      val = tf.nn.bias_add(val, self.b)
    
    if self.activation is not None:
      val = self.activation(val)

    self.add_loss(dkl_loss, inputs=True)

    if not context.executing_eagerly():
      # Set the static shape for the result since it might be lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      val.set_shape(self.compute_output_shape(inputs.shape))

    return val
  
  def compute_output_shape(self, input_shape):
    return input_shape

  def get_config(self):
    config = {
        'num_outputs' : self.num_outputs,
        'activation' : self.activation,
        'kernel_initializer' : self.kernel_initializer,
        'log_sigma2_initializer' : self.log_sigma2_initializer,
        'use_bias' : self.use_bias,
        'eps' : self.eps,
        'threshold' : self.threshold,
        'clip_alpha' : self.clip_alpha,
        'warmup':self.warmup
    }
    base_config = super(VarDropout, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))

@tf.function
def compute_log_sigma2(theta, log_alpha):
    return log_alpha + tf.math.log(tf.square(theta) + 1e-8)

@tf.function
def compute_log_alpha(theta, log_sigma2):
  return log_sigma2 - tf.math.log(tf.square(theta) + 1e-8)


class VarDropoutLeNetBlock(Layer):
    def __init__(self, activation = tf.keras.activations.relu):
        super(VarDropoutLeNetBlock, self).__init__()
        self.activation = activation
        self.dense_1 = Dense(300, activation = activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.var_dropout_1 = VarDropout(name='var_dropout1',activation = activation,kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.dense_2 = Dense(100, activation = activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.var_dropout_2 = VarDropout(name='var_dropout2',activation = activation,kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.dense_3 = Dense(100, activation = activation, kernel_regularizer=tf.keras.regularizers.l2(1e-3))
        self.var_dropout_3 = VarDropout(name='var_dropout3',activation = activation,kernel_regularizer=tf.keras.regularizers.l2(1e-3))

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.var_dropout_1(x)
        x = self.dense_2(x)
        x = self.var_dropout_2(x)
        x = self.dense_3(x)
        x = self.var_dropout_3(x)
        return x

    def get_config(self):
      config = {
        'dense_1' : self.dense_1,
        'var_dropout1' : self.var_dropout_1,
        'dense_2' : self.dense_2,
        'var_dropout2' : self.var_dropout_2,
        'dense_3' : self.dense_3,
        'var_dropout3' : self.var_dropout_3,
        'activation':self.activation
      }
      base_config = super(VarDropout, self).get_config()
      return dict(list(base_config.items()) + list(config.items()))

class DropoutLeNetBlock(Layer):
    def __init__(self, activation = tf.keras.activations.relu, rate = .2):
        super(DropoutLeNetBlock, self).__init__()
        self.activation = activation
        self.dense_1 = Dense(300, activation = activation)
        self.dropout_1 = Dropout(rate)
        self.dense_2 = Dense(100, activation = activation)
        self.dropout_2 = Dropout(rate)
        self.dense_3 = Dense(100, activation = activation)
        self.dropout_3 = Dropout(rate)

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.dropout_1(x)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)
        return x

class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def create_model(
                x_train, 
                initial_values, 
                num_labels, 
                encoder, 
                model_name,
                dropout_type = 'var',
                dropout_rate = .2):
  
  # Define Inputs
  model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
  
  #x = ConstantGausianDropoutGate(initial_values[i])(model_input)
  if dropout_type == 'var':
        x = Dense(300,kernel_regularizer=tf.keras.regularizers.l2(0.001))(model_input)
        x = VarDropout(kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = VarDropout(kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
        x = VarDropout(kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
  else:
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)
  model_out = Dense(num_labels,name='model_out')(x)
  
  # define model
  model = Model(model_input, model_out, name = model_name)
  model.summary()
  
  return model

def custom_train(model, x_train, y_train, optimizer, x_test,y_test):
  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  # Instantiate a loss
  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits=True,reduction=tf.keras.losses.Reduction.AUTO)

  # Instantiate an optimizer.
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
  val_accuracy = tf.keras.metrics.CategoricalAccuracy()



  # Prepare the validation dataset.
  # Reserve 5,000 samples for validation.
  x_val = x_train[-5000:]
  y_val = y_train[-5000:]
  x_train = x_train[:-5000]
  y_train = y_train[:-5000]
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(BATCH_SIZE)

  # prepare data  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

  @tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      logits = model(x_batch_train, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

      # Add kld layer losses created during this forward pass:
      kld_losses =  sum(model.losses) 
      kld_losses = kld_losses #/ float(x_train.shape[0])
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
  

  test_step(x_test, y_test)
  val_acc = val_accuracy.result()
  val_accuracy.reset_states()
  print("Test acc: %.4f" % (float(val_acc),))
    

def get_varparams_class_samples(predictions, y_test, num_labels):
  initial_thetas = []
  initial_log_sigma2s = []  

  for x in range(num_labels):
    targets = np.where(y_test == x)[0]
    sample = targets[np.random.randint(targets.shape[0])]
    initial_thetas.append(predictions[sample][0])
    initial_log_sigma2s.append(predictions[sample][1])
  
  initial_values = np.transpose(np.stack([initial_thetas,initial_log_sigma2s]))
  return initial_values

def get_varparams_class_means(predictions, y_test, num_labels):
  initial_thetas = []
  initial_log_sigma2s = []
  
  for x in range(num_labels):
    targets = predictions[np.where(y_test == x)[0]]
    means = np.mean(targets, axis = 0)
    initial_thetas.append(means[0])
    initial_log_sigma2s.append(means[1])

  initial_values = np.transpose(np.stack([initial_thetas,initial_log_sigma2s]))
  return initial_values


def loss(target_y, predicted_y):
  loss_fn = tf.keras.losses.categorical_crossentropy(from_logits=True)
  loss = loss_fn(target_y,predicted_y)

  return tf.reduce_mean(tf.square(target_y - predicted_y))

# Settings
DIMENSION = 784
EPOCHS = 100
intermediate_dim = 512
BATCH_SIZE = 100
latent_dim = 2
vae_epochs = 1

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-c", "--categorical",
                      default=True,
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
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch')


  # Create a callback that saves the model's weights
  checkpoint_path = "training\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=1)

  # load data
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]
  
  # Define encoder model.
  original_inputs = tf.keras.Input(shape=(DIMENSION,), name="encoder_input")
  x = Dense(intermediate_dim, activation="relu")(original_inputs)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()((z_mean, z_log_var))
  encoder = tf.keras.Model(inputs=original_inputs, outputs=[z,z_mean,z_log_var], name="encoder")

  # Define decoder model.
  latent_inputs = tf.keras.Input(shape=(latent_dim,), name="z_sampling")
  x = Dense(intermediate_dim, activation="relu")(latent_inputs)
  outputs = Dense(DIMENSION, activation="sigmoid")(x)
  decoder = tf.keras.Model(inputs=latent_inputs, outputs=outputs, name="decoder")

  # Define VAE model.
  outputs = decoder(z)
  if os.path.isfile(checkpoint_path):
    vae.load_weights(checkpoint_path)
  vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
  vae.summary()

  # Add KL divergence regularization loss.
  kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
  kl_loss_i = tf.identity(kl_loss)
  vae.add_loss(kl_loss_i)

  # Train
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())
  vae.fit(x_train, x_train, epochs=vae_epochs, batch_size=BATCH_SIZE)
  vae.save_weights(checkpoint_path)


  # gather predictions for the test batch
  predictions, _, _ = encoder.predict(x_test, batch_size=BATCH_SIZE) 

  if args.embedding_type == 'sample':
    initial_values = get_varparams_class_samples(predictions, y_test, num_labels)
  else:
    initial_values = get_varparams_class_means(predictions, y_test, num_labels)


  # create model under test
  model = create_model(x_train, initial_values, num_labels, encoder, model_name)
  
  loss_fn = tf.losses.CategoricalCrossentropy(from_logits = True)
  metrics = [keras.metrics.CategoricalAccuracy()]

  
  STEPS_PER_EPOCH = x_train.shape[0]//BATCH_SIZE
  lr_schedule = tf.keras.optimizers.schedules.InverseTimeDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*1000,
    decay_rate=1,
    staircase=False)

  tf.keras.optimizers.Adam(lr_schedule)

  # Train
  # use custom training loop to assist in debugging
  #custom_train(model, x_train, y_train, optimizer, x_test, y_test)
  
  # use graph training for speed
  model.compile(optimizer,loss = loss_fn, metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[sparsity_cb], validation_split=.05)

  # Add KL divergence regularization loss.
  kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
  kl_loss_i = tf.identity(kl_loss)
  vae.add_loss(kl_loss_i)

  #model accuracy on test dataset
  score = model.evaluate(x = x_test, y = y_test, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))