from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse
from datetime import datetime
# Using Base Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Reshape, GaussianDropout
from tensorflow.keras.layers import Concatenate, Softmax, Conv2D, MaxPooling2D, Flatten, Layer, Multiply, Add, Subtract, Average, Activation,BatchNormalization
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.python.eager import context
from tensorflow.keras.initializers import Initializer, TruncatedNormal, Identity, RandomNormal
from numpy import linalg as LA
#import tensorflow_model_optimization as tfmot
losses = tf.keras.losses

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from layers import VarDropout, ProbabilityDropout
import custom_train
import utils
#import keract
import time
import plaidml

callbacks = tf.keras.callbacks
K = tf.keras.backend

#from packaging import version

EPSILON = 1e-8
ALPHA = 8.
MEAN_EVAL = True



class Floor(Layer):
  def __init__( self,
              zero_point = None,
              mean_on_eval = MEAN_EVAL,

              **kwargs):
    super(Floor, self).__init__(**kwargs)
    self.zero_point = zero_point
    self.mean_on_eval = mean_on_eval


  def call(self, inputs, training = None):
    if not training and self.mean_on_eval:
      return inputs
    else:

      if self.zero_point is None:
        zero_point = tf.random.uniform([],0,.7)
      else:
        zero_point = self.zero_point

      condition = tf.less(tf.abs(inputs), zero_point)
      x = tf.where(condition,tf.zeros_like(inputs),inputs)

      return x

  def get_config(self):
    return {
      "zero_point" : zero_point,
      "mean_on_eval" : mean_on_eval
    }


def vd_loss(model):
  log_alphas = []
  fraction = 0.
  theta_logsigma2 = [layer.variables for layer in model.layers if 'vd_' in layer.name]
  for theta, log_sigma2, b in theta_logsigma2:
    log_alphas.append(tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + EPSILON),-ALPHA,ALPHA))

  return log_alphas

def negative_dkl(log_alpha):
  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1

  dkl = k1*tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5*tf.math.log1p(tf.exp(tf.math.negative(log_alpha))) + c
  return -tf.reduce_sum(dkl)


  def negative_dkl(self,theta,log_sigma2):
    # Constant values for approximating the kl divergence
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -k1

    log_alpha = tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + EPSILON),-ALPHA,ALPHA)
    dkl = k1*tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5*tf.math.log1p(tf.exp(tf.math.negative(log_alpha))) + c
    return -tf.reduce_sum(dkl)


def custom_train(model, x_train, y_train, optimizer, x_test,y_test, loss_fn):
  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  # Instantiate an optimizer.
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
  val_accuracy = tf.keras.metrics.CategoricalAccuracy()

  # Prepare the validation dataset.
  # Reserve 5,000 samples for validation.
  x_val = x_train[-3240:]
  y_val = y_train[-3240:]
  x_train = x_train[:-3240]
  y_train = y_train[:-3240]
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(BATCH_SIZE)

  # prepare data
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)


  #@tf.function
  def train_step(x, y, dkl_fraction):
    with tf.GradientTape() as tape:
      logits = model(x, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = tf.nn.softmax_cross_entropy_with_logits(y, logits)

      # Add kld layer losses created during this forward pass:
      if layer_losses:
        dkl_loss =  sum(model.losses)  # from layer losses
      else:
        log_alphas = vd_loss(model)
        dkl_loss = tf.add_n([negative_dkl(log_alpha=a) for a in log_alphas])
        dkl_fraction = dkl_fraction + .001
        dkl_fraction = tf.maximum(1.,dkl_fraction)
        dkl_loss = dkl_loss * step / 50000

      loss_value += dkl_loss

      tf.summary.scalar('dkl_loss_net', dkl_loss)
      tf.summary.scalar('dkl_fraction', dkl_fraction)
      tf.summary.scalar('dkl_loss_gross', dkl_loss )

      # Retrieve the gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(loss_value, model.trainable_weights)
      # Minimize the loss.
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metrics
      epoch_loss_avg.update_state(loss_value)
      epoch_accuracy.update_state(y_batch_train,logits)
    return loss_value

  #@tf.function
  def test_step(x, y):
      val_logits = model(x, training=False)
      val_accuracy.update_state(y, val_logits)

  dkl_fraction = tf.constant(0.)
  for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))

    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

      # Run the forward pass.
      loss_value = train_step(x_batch_train, y_batch_train, dkl_fraction)

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

class Pdf(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev):
    self.mean = tf.reduce_mean(mean)
    self.stddev = tf.reduce_mean(stddev)

  def __call__(self, shape, dtype=None):
    return tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype)

  def get_config(self):  # To support serialization
    return {'mean': self.mean, 'stddev': self.stddev}

class Kernel(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape, dtype=None):
    return tf.nn.softmax(tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype))

  def get_config(self):  # To support serialization
    return {'mean': self.mean, 'stddev': self.stddev}

class CGD(Layer):
  def __init__( self,
                initial_values,
                num_outputs = None,
                activation = tf.keras.activations.sigmoid,
                use_bias=True,
                zero_point = 1e-2,
                **kwargs):

    super(CGD, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.use_bias = use_bias
    self.zero_point = zero_point
    mean, var = initial_values


    self.initial_theta = initial_theta
    self.initial_log_sigma2 = tf.sqrt(initial_log_sigma2)

  def build(self, input_shape):
    kernel_shape = (input_shape[-1],self.num_outputs)

    self.kernel = self.add_weight(shape = kernel_shape,
                            initializer=tf.keras.initializers.TruncatedNormal(self.initial_theta,self.initial_log_sigma2),
                            trainable=False,
                            name="cgd-kernel")

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.keras.initializers.Constant(self.initial_theta),
                            trainable = False, name = 'cgd-bias')
    else:
      self.b = None
    self.built = True

  def call(self, inputs, training = None):
    val = tf.matmul(inputs, self.kernel)

    if self.use_bias:
      val = tf.nn.bias_add(val, self.b)

    if self.activation is not None:
      val = self.activation(val)

    if not context.executing_eagerly():
      val.set_shape(self.compute_output_shape(inputs.shape))

    return val

  def compute_output_shape(self, input_shape):
    return  (input_shape[0],self.num_outputs)

class PD(Layer):
  def __init__( self,
                initial_values,
                num_outputs = None,
                activation = None,
                zero_point = 1e-2,
                use_bias = True,
                **kwargs):
    super(PD, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.zero_point = zero_point
    self.use_bias = use_bias
    mean, var = initial_values
    self.mean = mean
    self.stddev = tf.sqrt(var)

  def build(self, input_shape):

    if self.num_outputs is None:
      self.num_outputs = input_shape[-1]

    kernel_shape = (input_shape[-1],self.num_outputs)

    self.w = self.add_weight('w', shape = kernel_shape,
                            initializer=tf.initializers.RandomNormal(),
                            trainable = True)

    if self.use_bias:
      self.b = self.add_weight('b', shape = (self.num_outputs,),
                              initializer=tf.initializers.RandomNormal(),
                              trainable=True)

  def call(self, inputs):
    f = tf.nn.sigmoid(tf.random.normal(tf.shape(self.w), mean=self.mean, stddev=self.stddev))
    x = tf.matmul(inputs,self.w * f)

    if self.activation is not None:
      x = self.activation(x)

    if self.use_bias:
      x = tf.nn.bias_add(x,self.b)

    return x

  def get_config(self):
    config = {
      'num_outputs': self.num_outputs,
      'activation':  activation,
      'zero_point': zero_point
    }
    base_config = super(PD, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


class SD(Layer):
  def call(self, inputs):
    logits, z_mean, z_log_var = inputs
    z_mean = tf.reshape(z_mean,[tf.shape(z_mean)[0],1])
    z_log_var = tf.reshape(z_log_var,[tf.shape(z_log_var)[0],1])
    epsilon = tf.random.normal(shape=tf.shape(logits))
    p =  z_mean + tf.exp(0.5 * z_log_var) * epsilon
    return p * logits

class Sampling(Layer):
    def call(self, inputs):
      z_mean, z_log_var = inputs
      epsilon = tf.random.normal(shape = tf.shape(z_mean))
      return z_mean + tf.math.exp(0.5 * z_log_var) * epsilon

def create_model(
                x_train,
                initial_values,
                num_labels,
                encoder,
                decoder,
                loss_fn ,
                model_type,
                encodings = None,
                dropout_rate = .2,
                classp = None):

  print('\n\n')
  if model_type == 'cgd_model':
    print('Building CGD Model')

    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name=model_type+'_input')
    y = []
    for i in range(num_labels):
      x = Dense(300,  activation = 'relu')(model_input)
      y.append(CGD([initial_values[i][0],initial_values[i][1]],300,activation=tf.nn.relu)(x))

    x = Concatenate(name=model_type+"concat")(y)
    x = Dense(300,  activation = 'relu', name=model_type+'_d1')(x)
    x = Dense(300,  activation = 'relu', name=model_type+'_d2')(x)

    model_out = Dense(num_labels, name=model_type+'_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'gatedmoe':
    print('Building gatedmoe Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    y = []
    for i in range(num_labels):
      x = Dense(300, kernel_initializer=tf.keras.initializers.RandomNormal(initial_values[i][0],initial_values[i][1]), activation='relu')(model_input)
      x = Dense(300,activation = tf.nn.relu)(x)
      x = Dense(300,activation = tf.nn.relu)(x)
      y.append(x)

    x = Concatenate()(y)
    x = Dense(200, activation = tf.nn.relu, )(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'gatedmoe_no_init':
    print('Building gatedmoe Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    y = []
    for i in range(num_labels):
      x = Dense(300, activation = 'relu', name = "Gate"+str(i))(model_input)
      x = Dense(300, activation = 'relu')(x)
      x = Dense(300, activation = 'relu')(x)
      y.append(x)

    x = Concatenate()(y)
    x = Dense(200, activation = tf.nn.relu, )(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model
  elif model_type == 'vdmoe':
    print('Building vdmoe Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    z_mean, z_log_var, z = encoder(model_input)
    y = []
    for i in range(num_labels):
      x = VarDropout(300)(model_input)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d1'+str(i))(x)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d2'+str(i))(x)
      y.append(x)

    x = Concatenate()(y)
    x = Dense(200)(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'vd_ref':
    print('Building vd_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='xin')
    x = VarDropout(300, name = _md.model_name+'_d1',activation = tf.keras.activations.relu)(model_input)
    x = VarDropout(300, name = _md.model_name+'_d2',activation = tf.keras.activations.relu)(x)
    x = VarDropout(300, name = _md.model_name+'_d3',activation = tf.keras.activations.relu)(x)

    model_out = Dense(num_labels, name ='xout')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'dense_ref':
    print('Building dense_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = Dropout(_md.in_zero_point)(model_input)
    x = Dense(300, activation='relu',name = model_type+'_d1')(x)
    #x = Dropout(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d2')(x)
    #x = Dropout(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d3')(x)
    x = Dropout(_md.zero_point)(x)

    model_out = Dense(num_labels, name = 'model_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'dense_gauss_ref':
    print('Building dense_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = GaussianDropout(_md.in_zero_point)(model_input)
    x = Dense(300, activation='relu',name = model_type+'_d1')(x)
    x = GaussianDropout(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d2')(x)
    x = GaussianDropout(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d3')(x)
    x = GaussianDropout(_md.zero_point)(x)

    model_out = Dense(num_labels, name = 'model_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'dense_floor':
    print('Building dense_floor Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = Floor(_md.in_zero_point)(model_input)
    x = Dense(300, activation='relu',name = model_type+'_d1', use_bias= _md.use_bias)(x)
    #x = Floor(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d2', use_bias= _md.use_bias)(x)
    #x = Floor(_md.zero_point)(x)
    x = Dense(300, activation='relu',name = model_type+'_d3', use_bias= _md.use_bias)(x)
    x = Floor(_md.zero_point)(x)

    model_out = Dense(num_labels, name = 'model_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'dropout_floor':
    print('Building dropout_floor')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = Dropout(_md.in_dropout)(model_input)
    x = Floor(_md.in_zero_point)(model_input)

    x = Dense(300, activation='relu', name = model_type+'_d1')(x)
    x = Dense(300, activation='relu', name = model_type+'_d2')(x)
    x = Dense(300, activation='relu', name = model_type+'_d3')(x)
    x = Dropout(_md.dropout)(x)
    x = Floor(_md.zero_point)(x)

    model_out = Dense(num_labels, name = 'model_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'pdf':
    print('Building pdf Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    x = Dense(300, activation =  'relu', name=_md.model_name+'_d1')(model_input)
    x = Dense(300, activation =  'relu', name=_md.model_name+'_d2')(x)
    x = Dense(300, activation =  'relu', name=_md.model_name+'_d3')(x)
    x = Dense(300, kernel_initializer=Pdf(initial_values[0],initial_values[1]), activation =  'relu', name=model_type+'_d1')(x)

    model_out = Dense(num_labels,name=model_type+'_output')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'classpd':
    print('Building Class Probability Dropout MLP Model')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    y = []
    x = Dense(300, activation='relu')(xin)
    x = Dropout(.2)(x)
    for i in range(num_labels):
      x = PD(initial_values[i], 300,activation=tf.nn.relu)(x)

      y.append(x)

    x = Concatenate()(y)
    x = Floor()(x)
    x = Dense(300, activation='relu')(x)
    x = Dropout(.2)(x)
    xout = Dense(10)(x)
    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'classpd_floor':
    print('Building Class Probability Dropout MLP Model')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    y = []
    x = Dense(300, activation='relu')(xin)
    x = Floor(.01)(x)
    for i in range(num_labels):
      x = PD(initial_values[i], 300,activation=tf.nn.relu)(x)
      x = Dense(300, activation='relu')(x)
      x = Floor(.3)(x)
      y.append(x)

    x = Concatenate()(y)
    x = Dense(300, activation='relu')(x)
    x = Floor(.3)(x)
    xout = Dense(10)(x)
    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'class_ref':
    print('Building Class Probability Dropout MLP Model')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    y = []
    x = Dense(300, activation='relu')(xin)
    x = Dropout(.2)(x)
    for i in range(num_labels):
      x = Dense(300, activation='relu')(x)
      y.append(x)

    x = Concatenate()(y)
    x = Dense(300, activation='relu')(x)
    x = Dropout(.4)(x)
    xout = Dense(10)(x)
    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'pd':
    print('Building Probability Dropout MLP Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='model_input')

    x = PD(initial_values, 300, name= _md.model_name+'_pd1')(model_input)
    x = Activation('relu')(x)

    x = PD(initial_values, 300, name= _md.model_name+'_pd2')(x)
    x = Activation('relu')(x)

    x = PD(initial_values, 300, name= _md.model_name+'_pd3')(x)
    x = Activation('relu')(x)

    model_out = Dense(num_labels,name= _md.model_name+"_output")(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'sampling_dropout':
    print('Building sampling_dropout Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(model_input)
    z_mean, z_var = tf.nn.moments(z,axes = -1)

    x = Dense(300,activation = 'relu',name = _md.model_name+'_d1')(model_input)
    x = SD()([x,z_mean,z_var])
    x = Dense(300, activation = 'relu',name = _md.model_name+'_d2')(x)
    #x = SD()([x,z_mean,z_var])
    x = Dense(300, activation = 'relu',name = _md.model_name+'_d3')(x)
    x = SD()([x,z_mean,z_var])

    model_out = Dense(num_labels, name=_md.model_name+"_out")(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'mlp-stack':
    print('Building Encoder-MLP Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    z_mean, z_log_var, z = encoder(model_input)

    x = Dense(300,activation = 'relu')(z)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    model_out = Dense(num_labels, name=_md.model_name)(x)
    model = Model(model_input, model_out, name =_md.model_name  )


    if _md.kld_loss:
      kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
      kl_loss = tf.reduce_sum(kl_loss, axis=-1)
      kl_loss *= -0.5
      model.add_loss(kl_loss)

    return model

  elif model_type == 'mlp-stack2':
    print('Building Encoder-MLP Stack2')
    input_dim = output_dim = x_train.shape[-1]

    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    z_mean, z_log_var, z = encoder(model_input)
    decoder_out = decoder(z)

    x = Dense(300,activation = 'relu')(z)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    model_out = Dense(num_labels, name=_md.model_name)(x)

    model = Model(model_input, [model_out,decoder_out], name =_md.model_name  )

    reconstruction_loss = losses.mean_squared_logarithmic_error(model_input, decoder_out)
    reconstruction_loss *= model_input

    kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # Add loss to model
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    model.compile(_md.optimizer,
          loss = [tf.keras.losses.CategoricalCrossentropy(from_logits = True),vae_loss],
          metrics = _md.metrics,
          experimental_run_tf_function = False)

    return model

  elif model_type == 'conv_ref':
    print('Building Reference CONV model')

    print('Building Reference Conv Model with Dropout')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    if dataset != 'cifar10':
      x = Reshape(target_shape = (28, 28, 1))(xin)
    else:
      x = Reshape(target_shape = (32, 32, 3))(xin)

    x = Conv2D(filters = 96, kernel_size=(3, 3), activation='relu')(x)
    x = Dropout(_md.in_dropout)(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation='relu', strides = 2)(x)
    x = Dropout(0.5)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu',activity_regularizer=tf.keras.regularizers.L2(.001))(x)
    x = Dropout(_md.dropout_rate)(x)
    xout = Dense(num_labels)(x)

    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'conv_gaussian_ref':
    print('Building Reference CONV model')

    print('Building Reference Conv Model with Dropout')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    if dataset != 'cifar10':
      x = Reshape(target_shape = (28, 28, 1))(xin)
    else:
      x = Reshape(target_shape = (32, 32, 3))(xin)

    x = Conv2D(filters = 96, kernel_size=(3, 3), activation='relu')(x)
    x = GaussianDropout(0.2)(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation='relu')(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation='relu', strides = 2)(x)
    x = GaussianDropout(0.5)(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu')(x)
    xout = Dense(num_labels)(x)

    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'conv_floor':
    print('Building Conv Model with Floor')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    if dataset != 'cifar10':
      x = Reshape(target_shape = (28, 28, 1))(xin)
    else:
      x = Reshape(target_shape = (32, 32, 3))(xin)

    x = Conv2D(filters = 96, kernel_size=(3, 3), activation=tf.nn.relu)(x)
    #x = Floor(_md.in_dropout)(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation=tf.nn.relu)(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation=tf.nn.relu, strides = 2)(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation=tf.nn.relu)(x)
    x = Floor(_md.zero_point)(x)

    xout = Dense(num_labels)(x)
    model = Model(xin, xout, name = _md.model_name)
    return model


  elif model_type == 'conv_dropfloor':
    print('Building Conv Model with Floor')
    xin = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    if dataset != 'cifar10':
      x = Reshape(target_shape = (28, 28, 1))(xin)
    else:
      x = Reshape(target_shape = (32, 32, 3))(xin)

    x = Conv2D(filters = 96, kernel_size=(3, 3), activation=tf.nn.leaky_relu)(x)

    x = Dropout(_md.in_dropout)(x)
    #x = Floor(_md.in_zero_point)(x)

    x = Conv2D(filters=192, kernel_size=(3,3), activation=tf.nn.leaky_relu)(x)
    x = MaxPooling2D((2,2))(x)
    x = Conv2D(filters=192, kernel_size=(3,3), activation=tf.nn.leaky_relu, strides = 2)(x)
    x = MaxPooling2D((2,2))(x)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation=tf.nn.leaky_relu)(x)

    x = Dropout(_md.dropout_rate)(x)
    x = Floor(_md.zero_point)(x)

    xout = Dense(num_labels)(x)

    model = Model(xin, xout, name = _md.model_name)
    return model

  elif model_type == 'smconv_ref':
    print('Building Reference small conv Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = Reshape(target_shape = (28, 28, 1))(i)
    x = Conv2D(filters = num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Conv2D(filters = num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = Floor(0.4)(x)
    x = Flatten()(x)
    x = Dense(10)(x)

    model = Model(i, x, name = _md.model_name)
    return model

  elif model_type == 'smconv_stack':
    print('Building small conv_stack Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(i)
    z = Reshape(target_shape=(4,4,1))(z)
    z = Flatten()(z)
    x = Reshape(target_shape = (28, 28, 1))(i)
    x = Flatten()(x)
    x = Concatenate()([x,z])

    x = Conv2D(filters = num_labels, kernel_size = (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    x = Dense(10)

    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  else:
    print('Building Reference Lenet300_100_100 Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)

    model_out = Dense(num_labels, name=model_type +'_out')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model


def create_vae(x_train):
    input_dim = output_dim = x_train.shape[-1]

    # Define encoder model.
    inputs = tf.keras.Input(shape=(input_dim,), name="encoder_input")
    x = Dense(_vae.intermediate,name=_vae.model_type +"_enc_d1")(inputs)
    x = Activation(_vae.act)(x)
    z_mean = Dense(_vae.latent, name=_vae.model_type +"_z_mean")(x)
    z_log_var = Dense(_vae.latent, name=_vae.model_type +"_z_log_var")(x)

    if _vae.model_type == 'ref':
      z = Sampling(name=_vae.model_type +"_z")([z_mean, z_log_var])
    else:

      z = VarDropout(_vae.latent, name=_vae.model_type +"_zvd")(x)
      x = Activation(_vae.act)(x)
    encoder = Model(inputs, outputs=[z_mean, z_log_var, z], name=_vae.enc_name)
    encoder.summary()

    # Define decoder model.
    latent_in = tf.keras.Input(shape=(_vae.latent,), name="latent_in")
    x = Dense(_vae.intermediate, name="decoder_d1")(latent_in)
    x = Activation(_vae.act)(x)
    outputs = Dense(input_dim, name='decoder_output', activation='sigmoid')(x)
    decoder = Model(latent_in, outputs, name=_vae.dec_name)
    decoder.summary()

    # Define VAE model
    outputs = decoder(encoder(inputs)[2])
    vae = tf.keras.Model(inputs, outputs, name=_vae.vae_name)
    vae.summary()

    # Add loss
    if   _vae.loss == 'mse':
      reconstruction_loss = losses.mean_squared_error(inputs, outputs)
    elif _vae.loss == 'msle':
      reconstruction_loss = losses.mean_squared_logarithmic_error(inputs, outputs)
    elif _vae.loss == 'bce':
      reconstruction_loss = losses.binary_crossentropy(inputs, outputs)
    elif _vae.loss == 'cce':
      reconstruction_loss = losses.categorical_crossentropy(inputs, outputs)

     # Scale the loss
    reconstruction_loss *= input_dim

    # kld regularization
    if _vae.global_kld:
      kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      kl_loss = tf.reduce_sum(kl_loss, axis=-1)
      kl_loss *= -0.5

    else:
      kl_loss = 0

    # Add loss to model
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    # enc.compile(_vae.optimizer)
    # dec.compile(_vae.optimizer)
    vae.compile(_vae.optimizer)

    return vae, encoder, decoder


# Training settings
EPOCHS = 20
BATCH_SIZE = 128
VAE_EPOCHS = 10
CUSTOM_TRAIN = False
BUILD_VAE = False
dataset = 'mnist'
layer_losses = True

##########################################################################
# Settings
#

class VaeSettings(object):
  model_type = 'ref'
  global_kld = True

  loss = 'msle'
  intermediate = 512
  latent = 16
  optimizer = tf.keras.optimizers.Adam()
  act = 'relu'

  if global_kld:
    reg = 'g'
  else:
    reg = 'l'

  vae_name = "vae-{}-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act, dataset)
  enc_name = "enc-{}-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act, dataset)
  dec_name = "dec-{}-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act, dataset)
_vae = VaeSettings()

class ModelSettings(object):
  model_type = 'classpd'
  zero_point = None
  in_zero_point = None
  dropout_rate = .2
  in_dropout = .2


  kld_loss = False
  enc_trainable = True
  use_bias = True


  loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
  optimizer = tf.keras.optimizers.Adam(tf.keras.optimizers.schedules.ExponentialDecay(
    .001,
    decay_steps=3000,
    decay_rate=.96,
    staircase=False
  ))
  metrics = [keras.metrics.CategoricalAccuracy()]
  act = 'relu'

  if enc_trainable:
    t = 'trn'
  else:
    t = 'lock'

  if kld_loss:
    reg = 'kld'
  else:
    reg = 'nreg'

  if MEAN_EVAL:
    me = 'mean'
  else:
    me = 'floor'

  model_name = "{}-{}-{}-{}-{}".format(model_type, dataset, zero_point, in_zero_point, me)
_md = ModelSettings()

save_dir = os.path.join(os.getcwd(), 'saved_models')
log_dir = "logs/fit/" + _md.model_name + datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__':

  (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_dataset(dataset,True)


  ####################################################################
  # vae
  #
  # Callbacks
  #sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch')
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir,
                                                histogram_freq = 2,
                                                write_graph=False,
                                                write_images=False,
                                                update_freq = 'epoch',
                                                profile_batch = [2,10]
                                                )

  # build
  if BUILD_VAE:
    vae, enc, dec = create_vae(x_train)
    # fit
    vae.fit(x_train, x_train,
            epochs = VAE_EPOCHS,
            batch_size = BATCH_SIZE,
            validation_split = .1,
            shuffle = True,
            callbacks =[tensorboard_cb],
            verbose=1
          )

    vae.save('models/'+ _vae.vae_name)
    enc.save('models/'+ _vae.enc_name)
    dec.save('models/'+ _vae.dec_name)

    # plot
    # plot_model(vae, to_file= "plots/" +_vae.vae_name + '.png', show_shapes=True, show_layer_names=False)
    # plot_model(enc, to_file= "plots/" +_vae.enc_name + '.png', show_shapes=True, show_layer_names=False)
    # plot_model(dec, to_file= "plots/" +_vae.dec_name + '.png', show_shapes=True, show_layer_names=False)

    # latent_fig = utils.plot_encoding(enc,
    #                 [x_test, y_test],
    #                 batch_size=BATCH_SIZE,
    #                 model_name=_vae.model_type)

    # utils.plot_reconstruction(dec, x_test)

  else:
    vae = tf.keras.models.load_model('models/'+ _vae.vae_name)
    enc = tf.keras.models.load_model('models/'+ _vae.enc_name)
    dec = tf.keras.models.load_model('models/'+ _vae.dec_name)

  _, _, encodings = enc.predict(x_test, batch_size = BATCH_SIZE)
  classp = utils.get_classp(encodings, y_test, num_labels)

  # calculate class means
  initial_values = utils.get_varparams_class_params(encodings, y_test, num_labels )

  ####################################################################
  # model
  #
  model = create_model(
                x_train,
                initial_values,
                num_labels,
                enc, dec,
                _md.loss_fn ,
                _md.model_type,
                encodings = encodings,
                dropout_rate = _md.zero_point,
                classp = classp)
  model.summary()

  model.compile(_md.optimizer,
            loss = _md.loss_fn,
            metrics = _md.metrics,
            experimental_run_tf_function = False)

  model.summary()
#  plot_model(model, to_file = "plots/"+_md.model_name +'.png', show_shapes = False)

  ####################################################################
  # Train
  #
  if CUSTOM_TRAIN:
    custom_train(model, x_train, y_train, _md.optimizer, x_test, y_test, _md.loss_fn)

  else:

    if not _md.enc_trainable:
      enc.trainable = False

    file_writer = tf.summary.create_file_writer(log_dir)
    file_writer.set_as_default()

    def vd_loss(model):
      log_alphas = []
      fraction = 0.
      theta_logsigma2 = [layer.variables for layer in model.layers if 'vd_' in layer.name]
      for theta, log_sigma2, b in theta_logsigma2:
        log_alphas.append(tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + EPSILON),-ALPHA,ALPHA))

      return log_alphas

    class SparseCallback(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs=None):
        total_w = 0.
        total_b = 0.
        w_non_zeros = 0.
        b_non_zeros = 0.
        fp_zero = 1e-3
        layers = [layer.variables for layer in self.model.layers if 'dense' in layer.name]

        if _md.use_bias:
          for kweights, biases in layers:
            w_non_zeros += tf.math.count_nonzero(kweights).numpy()
            b_non_zeros += tf.math.count_nonzero(biases).numpy()
            total_w += (tf.cast(tf.reduce_prod(tf.shape(kweights)), tf.float32)).numpy()
            total_b += (tf.shape(biases)[0]).numpy()

          ksparsity = 1. - w_non_zeros/total_w
          bsparsity = 1. - b_non_zeros/total_b
          tf.summary.scalar('kernel non zero weights', data=w_non_zeros, step = epoch)
          tf.summary.scalar('bias non zero weights', data=b_non_zeros, step = epoch)
          tf.summary.scalar('kernel sparsity', data=ksparsity, step = epoch)
          tf.summary.scalar('bias sparsity', data=bsparsity, step = epoch)


        else:
          for kweights in layers:
            w_non_zeros += tf.math.count_nonzero(kweights).numpy()
            total_w += (tf.cast(tf.reduce_prod(tf.shape(kweights)), tf.float32)).numpy()

          ksparsity = 1. - w_non_zeros/total_w
          tf.summary.scalar('kernel non zero weights', data=w_non_zeros, step = epoch)
          tf.summary.scalar('kernel sparsity', data=ksparsity, step = epoch)


    model.fit(x_train, y_train,
              epochs = EPOCHS,
              batch_size = BATCH_SIZE,
              validation_data=(x_test, y_test_cat),
              shuffle = True,
              callbacks=[tensorboard_cb],
              verbose=1
              )


  ####################################################################
  # Evaluate
  #
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE, callbacks=[tensorboard_cb])
  print('\n')
  print(str(_md.model_name) + ' Model Test Loss : ', score[0])
  print(str(_md.model_name) + ' Test Accuracy : %.4f%%' % (100.0 * score[1]))

