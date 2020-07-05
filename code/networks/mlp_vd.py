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
from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
#from layers.vd import VarDropout, ConstantGausianDropout
import utils


#######################################################
# vd

'''
input [10 classes]
mean std from vae ->
gate(program gate with a threshold that prefers a particular class)
Mixture of expert, Outrageously large mixture of experts



'''

EPSILON = 1e-8

class ConstantGausianDropoutGate(Layer):
  def __init__( self,
                initial_values,
                num_outputs = None,
                activation = None,
                use_bias=False,
                zero_point = 1e-2,
                **kwargs):

    super(ConstantGausianDropoutGate, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.use_bias = use_bias
    self.zero_point = zero_point

    # unpack variational parameters, and extrapolate the number of classes for the lookup  
    initial_theta, initial_log_sigma2 = initial_values

    # define static lookups for pre-calculated datasets
    self.initial_theta = initial_theta
    self.initial_log_sigma2 = initial_log_sigma2

  def build(self, input_shape):
    if self.num_outputs is None:
      self.num_outputs = input_shape[-1]

    kernel_shape = (input_shape[-1],self.num_outputs)

    self.kernel = self.add_weight(shape = kernel_shape,
                            initializer=tf.keras.initializers.RandomNormal(self.initial_theta,self.initial_log_sigma2),
                            trainable=False)
    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.keras.initializers.constant_initializer(self.initial_theta),
                            trainable = False)
    else:
      self.b = None

  def call(self, inputs, training = None):
    val = tf.matmul(inputs,self.kernel)     

    if self.activation is not None:
      val = self.activation(val)
    
    # # push values that are close to zero, to zero, promotes sparse models which are more efficient
    # condition = tf.less(val,self.zero_point)
    # val = tf.where(condition,tf.zeros_like(val),val)

    if not context.executing_eagerly():
      # Set the static shape for the result since it might lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      val.set_shape(self.compute_output_shape(inputs.shape))
    
    return val
      
  def compute_output_shape(self, input_shape):
    return  (input_shape[0],self.num_outputs)
  

class VarDropout(Layer):
  def __init__(self,
               num_outputs = None,
               activation = None,
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

    # if self.use_bias:
    #   self.b = self.add_weight(shape = (self.num_outputs,),
    #                         initializer = tf.constant_initializer(0.),
    #                         trainable = True)
    # else:
    #   self.b = None

    
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
      
      # # Constants
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

    # if self.use_bias:
    #   val = tf.nn.bias_add(val, self.b)
    
    if self.activation is not None:
      val = self.activation(val)

    self.add_loss(dkl_loss, inputs=True)

    if not context.executing_eagerly():
      # Set the static shape for the result since it might be lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      val.set_shape(self.compute_output_shape(inputs.shape))

    return val
  
  def compute_output_shape(self, input_shape):
    return (input_shape[0],self.num_outputs)

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

class CGDDropoutLeNetBlock(Layer):
    def __init__(self, initial_values,  activation = None, rate = .2):
        super(CGDDropoutLeNetBlock, self).__init__()
        self.activation = activation
        self.rate = rate
        self.dropout_1 = ConstantGausianDropoutGate(initial_values, activation = activation)
        self.dense_2 = Dense(100, activation = activation)
        self.dropout_2 = Dropout(rate)
        self.dense_3 = Dense(100, activation = activation)
        self.dropout_3 = Dropout(rate)

    def call(self, inputs):
        x = self.dropout_1(inputs)
        x = self.dense_2(x)
        x = self.dropout_2(x)
        x = self.dense_3(x)
        x = self.dropout_3(x)
        return x

class SamplingDropout(Layer):
    def call(self, inputs):
        logits, z_mean,z_log_var = inputs        
        z_mean = tf.reshape(z_mean,[tf.shape(z_mean)[0],1])
        z_log_var = tf.reshape(z_log_var,[tf.shape(z_log_var)[0],1])
        epsilon = tf.random.normal(shape=tf.shape(logits))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

class Sampling(Layer):
    def call(self, inputs):
      z_mean, z_log_var = inputs
      epsilon = tf.random.normal(shape=tf.shape(z_mean))
      return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon

def create_model(
                x_train, 
                initial_values, 
                num_labels, 
                encoder, 
                dropout_type = 'preencoder',
                dropout_rate = .2):
  
  # Define Inputs
  model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
  if dropout_type == 'var':
    z,_,_ = encoder(model_input)
    x = Dense(300,kernel_regularizer=tf.keras.regularizers.l2(0.001))(model_input)
    x = VarDropout()(x)
    x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = VarDropout()(x)
    x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)
    x = VarDropout()(x)

  elif dropout_type == 'cgd':
    x = Dense(300)(model_input)
    y = []
    for i in range(num_labels):
      y.append(CGDDropoutLeNetBlock(initial_values[i], activation = tf.nn.sigmoid)(x))      
      #y.append(ConstantGausianDropoutGate(initial_values[i], activation = tf.nn.sigmoid)(x))
    x = Concatenate()(y)
    x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001))(x)

  elif dropout_type == 'preencoder':
    z,_,_ = encoder(model_input)
    z_mean = z[:,0]
    z_log_var = z[:,1]
    x = Dense(300)(model_input)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100)(x)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100)(x)
    
    model_out = Dense(num_labels,name=dropout_type)(x)
    model = Model(model_input, model_out, name = dropout_type)
    model.summary()
  

    return model
  
  elif dropout_type == 'vae':
    z,_,_ = encoder(model_input)
    vae_out = decoder(z)    
    x = Dense(300)(model_input)

  elif dropout_type == 'conv':
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    model_out = Dense(num_labels,name=dropout_type)(x)
    model = Model(model_input, model_out, name = dropout_type)
    model.summary()
    return model

  elif dropout_type == 'conv_cgd':
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=12, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    model_out = Dense(num_labels,name=dropout_type)(x)
    model = Model(model_input, model_out, name = dropout_type)
    model.summary()
    return model
    
  else:   
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)

  model_out = Dense(num_labels,name=dropout_type)(x)
  model = Model(model_input, model_out, name = dropout_type)
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

  #@tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      logits = model(x_batch_train, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = loss_fn(y_batch_train, logits)

      # Add kld layer losses created during this forward pass:
      vdloss(model)
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

def vdloss(model):
  log_alphas = []
  theta_logsigma2 = [layer.variables for layer in model.layers if 'var_dropout' in layer.name]
  for theta, log_sigma2 in theta_logsigma2:
    log_alphas.append(compute_log_alpha(theta, log_sigma2))
  return theta_logsigma2


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
original_dim = 784
EPOCHS = 30
intermediate_dim = 512
BATCH_SIZE = 128
latent_dim = 2
vae_epochs = 10

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
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch')


  # Create a callback that saves the model's weights
  checkpoint_path = "training\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=0)

  # load data
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]
  
  # Define encoder model.
  original_inputs = tf.keras.Input(shape=(original_dim,), name="encoder_input")
  x = Dense(intermediate_dim, activation="relu")(original_inputs)
  z_mean = Dense(latent_dim, name="z_mean")(x)
  z_log_var = Dense(latent_dim, name="z_log_var")(x)
  z = Sampling()([z_mean, z_log_var])
  encoder = tf.keras.Model(inputs=original_inputs, outputs=[z_mean,z_log_var,z], name="encoder")

  # Define decoder model.
  latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
  x = Dense(intermediate_dim, activation='relu')(latent_inputs)
  outputs = Dense(original_dim, activation='sigmoid')(x)
  decoder = Model(latent_inputs, outputs, name='decoder')
  decoder.summary()

  # Define VAE model.
  outputs = decoder(encoder(original_inputs)[2])
  vae = tf.keras.Model(inputs=original_inputs, outputs=outputs, name="vae")
  vae.summary()

  # Add KL divergence regularization loss.
  reconstruction_loss = tf.keras.losses.mean_squared_error(original_inputs, outputs)
  reconstruction_loss *= original_dim
  
  kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
  kl_loss = tf.reduce_sum(kl_loss, axis=-1)
  kl_loss *= -0.5

  # add loss to model
  vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
  vae.add_loss(vae_loss)

  # Train
  optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
  vae.compile(optimizer)
  #vae.fit(x_train, x_train, epochs=vae_epochs, batch_size=BATCH_SIZE)
  
  # utils.plot_encoding(encoder,
  #               [x_test, y_test],
  #               batch_size=BATCH_SIZE,
  #               model_name="vae_mlp")

  #encoder.trainable = False

  # gather predictions for the test batch
  predictions, _, _ = encoder.predict(x_test, batch_size=BATCH_SIZE) 

  if args.embedding_type == 'sample':
    initial_values = get_varparams_class_samples(predictions, y_test, num_labels)
  else:
    initial_values = get_varparams_class_means(predictions, y_test, num_labels)

  # create model under test
  model = create_model(x_train, initial_values, num_labels, encoder, dropout_type='var')
  
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
  custom_train(model, x_train, y_train, optimizer, x_test, y_test)
  
  # use graph training for speed
  model.compile(optimizer,loss = loss_fn, metrics=['accuracy'])
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_cb], validation_split=.05)
  plot_model(model,to_file= 'plots\\mlp_vd.png')


  #model accuracy on test dataset
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))