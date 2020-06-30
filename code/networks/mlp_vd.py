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
import tensorflow_probability as tfp
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
               log_sigma2_initializer = None,
               use_bias=True,
               eps=EPSILON,
               threshold=3.,
               clip_alpha=8.,
               warmup = False,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.log_sigma2_initializer = log_sigma2_initializer
    self.use_bias = use_bias
    self.eps = eps
    self.threshold = threshold
    self.clip_alpha = clip_alpha
    self.warmup = warmup

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
                            trainable=True)

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.constant_initializer(0.),
                            trainable = True)      
    else:
      self.b = None
    
  def call(self, inputs, training = None):

    if training:
      val = matmul_train (inputs, (self.theta, self.log_sigma2))
    else:
      val = matmul_eval (inputs, (self.theta, self.log_sigma2), threshold = self.threshold)

    if self.use_bias:
      val = tf.nn.bias_add(val, self.b)
    
    if self.activation is not None:
      val = self.activation(val)

    #compute element-wise dkl
    log_alpha = compute_log_alpha((self.theta, self.log_sigma2), self.eps, self.clip_alpha)
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -0.63576
    eltwise_dkl = k1 * tf.nn.sigmoid(k2 + k3*log_alpha) -0.5 * tf.math.log1p(tf.exp(-log_alpha)) + c

    dkl_loss = -tf.reduce_sum(eltwise_dkl)
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

def matmul_eval(
      x,
      variational_params,
      threshold=3.0,
      eps=EPSILON):
    theta, log_sigma2 = variational_params

    # Compute the weight mask by thresholding on
    # the log-space alpha values
    log_alpha = compute_log_alpha(variational_params, eps=eps, value_limit=None)
    weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

    return tf.matmul(x,theta * weight_mask)

def matmul_train(
      x,
      variational_params,
      clip_alpha=None,
      eps=EPSILON):
    theta, log_sigma2 = variational_params
    log_alpha = compute_log_alpha(variational_params, eps, clip_alpha)
    
    # Compute the mean and standard deviation of the distributions over the
    # activations
    mu = tf.matmul(x, theta)
    std = tf.sqrt(tf.matmul(tf.square(x),tf.exp(log_alpha)*tf.square(theta)) + eps)
    return mu + std * tf.random.normal(tf.shape(mu))

def compute_log_sigma2(theta,log_alpha, eps=EPSILON):
    return log_alpha + tf.math.log(tf.square(theta) + eps)

def compute_log_alpha(variational_params, eps=EPSILON, value_limit=8.):
  theta, log_sigma2 = variational_params
  log_alpha = log_sigma2 - tf.math.log(tf.square(theta) + eps)

  if value_limit is not None:
    return tf.clip_by_value(log_alpha, -value_limit, value_limit)
  return tf.identity(log_alpha)

def get_scaler( step,
            start_reg_ramp_up=0.,
            end_reg_ramp_up=100.,
            reg_scalar = 1):

  current_step_reg = tf.cast(tf.cast(step,tf.float32) - start_reg_ramp_up, tf.float32)
  current_step_reg = tf.maximum(0.0, current_step_reg)

  fraction_ramp_up_completed = tf.minimum(
      current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)
  fraction = tf.minimum(current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)
  
  return fraction * reg_scalar

class VarDropoutLeNetBlock(Layer):
    def __init__(self, activation = tf.keras.activations.relu):
        super(VarDropoutLeNetBlock, self).__init__()
        self.activation = activation
        self.dense_1 = Dense(300, activation = activation)
        self.var_dropout_1 = VarDropout(name='var_dropout1')
        self.dense_2 = Dense(100, activation = activation)
        self.var_dropout_2 = VarDropout(name='var_dropout2')
        self.dense_3 = Dense(100, activation = activation)
        self.var_dropout_3 = VarDropout(name='var_dropout3')

    def call(self, inputs):
        x = self.dense_1(inputs)
        x = self.var_dropout_1(x)
        x = self.dense_2(x)
        x = self.var_dropout_2(x)
        x = self.dense_3(x)
        x = self.var_dropout_3(x)
        return x

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
    x = VarDropoutLeNetBlock()(model_input)
  else:
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)
  model_out = Dense(num_labels, activation="softmax",name='model_out')(x)
  
  # define model
  model = Model(model_input, model_out, name = model_name)
  model.summary()
  
  return model

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


# Settings
DIMENSION = 784
EPOCHS = 10
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
  metric = [keras.metrics.CategoricalAccuracy(),keras.metrics.Mean() ]
  
  # Train
  # use custom training loop to assist in debugging
  custom_train(model, x_train, y_train)
  
  # use graph training for speed
  model.compile('adam',loss = loss_fn, metrics=[metric])
  #model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_callback])

  # model accuracy on test dataset
  score = model.evaluate(x_test, y_test, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))