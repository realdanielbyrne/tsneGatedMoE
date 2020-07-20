from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse
from datetime import datetime
# Using Base Keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input,Dense, Dropout, Reshape
from tensorflow.keras.layers import Concatenate, Softmax, Conv2D, MaxPooling2D, Flatten, Layer, Multiply, Add, Subtract, Average
from tensorflow.keras import losses
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import mnist
from tensorflow.python.eager import context
from tensorflow.keras.initializers import Initializer, TruncatedNormal, Identity, RandomNormal
from numpy import linalg as LA
import tensorflow_model_optimization as tfmot
losses = tf.keras.losses

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from layers import VarDropout, ProbabilityDropout
import custom_train
import utils
import keract
import time
callbacks = tf.keras.callbacks
K = tf.keras.backend

from packaging import version

EPSILON = 1e-8

class Pdf(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev):
    self.mean = mean
    self.stddev = stddev

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
                use_bias=False,
                zero_point = 1e-2,
                **kwargs):

    super(CGD, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.use_bias = use_bias
    self.zero_point = zero_point
    initial_theta, initial_log_sigma2 = initial_values

    # define static lookups for pre-calculated datasets
    self.initial_theta = initial_theta
    self.initial_log_sigma2 = initial_log_sigma2

  def build(self, input_shape):
    kernel_shape = (input_shape[-1],self.num_outputs)
    
    self.kernel = self.add_weight(shape = kernel_shape,
                            #initializer= Pdf(self.initial_theta,self.initial_log_sigma2),
                            initializer=tf.keras.initializers.RandomNormal(self.initial_theta,self.initial_log_sigma2),
                            #regularizer=tf.keras.regularizers.L1L2(0, 0.001),
                            trainable=True,
                            name="cgd-kernel")
                            
    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.keras.initializers.Constant(self.initial_theta),
                            trainable = True, name = 'cgd-bias')
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

class PGD(Layer):
  def __init__( self,
                p,
                num_outputs = None,
                activation = None,
                zero_point = 1e-2,
                **kwargs):
    super(PGD, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.zero_point = zero_point
    self.p = p

  def build(self, input_shape):
    kernel_shape = (input_shape[-1],self.num_outputs)
    p = self.p
    num_outputs = self.num_outputs
    s = tf.keras.activations.sigmoid(p)
    p_range = [K.min(s),K.max(s)]
    pfilt = tf.nn.softmax(tf.cast(tf.histogram_fixed_width(s, p_range, nbins = num_outputs),dtype=tf.float32))    
    pfilt2 = tf.transpose(tf.reshape(pfilt,(-1,1)))
    filt = tf.repeat(pfilt2, repeats = input_shape[-1], axis = 0)
    self.filter = tf.Variable(filt, trainable = True)

  def call(self, inputs):
    y = tf.matmul(inputs,self.filter)
    return y

class SD(Layer):
    def call(self, inputs):
      logits, z_mean, z_log_var = inputs        
      z_mean = tf.reshape(z_mean,[tf.shape(z_mean)[0],1])
      z_log_var = tf.reshape(z_log_var,[tf.shape(z_log_var)[0],1])
      epsilon = tf.random.normal(shape=tf.shape(logits)) 
      p =  tf.nn.softmax(z_mean + tf.exp(0.5 * z_log_var) * epsilon)
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
                loss_fn ,
                model_type,
                predictions = None,                
                dropout_rate = .2):
  
  print('\n\n') 
  if model_type == 'cgd_model':
    print('Building CGD Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name=model_type+'_input') 
    y = []
    for i in range(num_labels):  
      y.append(Dense(300,kernel_initializer=tf.keras.initializers.RandomNormal(initial_values[i][0],initial_values[i][1]),activation='relu')(model_input))

    x = Concatenate(name=model_type+"concat")(y)
    x = Dense(300,  activation = 'relu', name=model_type+'_d1')(x)
    x = Dense(300,  activation = 'relu', name=model_type+'_d2')(x)

    model_out = Dense(num_labels, name=model_type+'_output')(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'gatedmoe':
    print('Building gatedmoe Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data') 
    
    y = []
    for i in range(num_labels):        
      x = Dense(300,kernel_initializer=tf.keras.initializers.RandomNormal(initial_values[i][0],initial_values[i][1]),activation='relu')(model_input)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d1'+str(i))(x)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d2'+str(i))(x)
      y.append(x)

    x = Concatenate()(y)    
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'vdmoe':
    print('Building vdmoe Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data') 
    
    y = []
    for i in range(num_labels):        
      x = VarDropout(300, name = model_type+'_vd1'+str(i),activation = tf.keras.activations.relu)(model_input)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d1'+str(i))(x)
      x = Dense(100,  activation = tf.nn.relu, name = model_type+'_d2'+str(i))(x)
      y.append(x)

    x = Concatenate()(y)    
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'vd_ref':
    print('Building vd_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name=model_type + '_in') 
    x = VarDropout(300, name = model_type+'_d1',activation = tf.keras.activations.relu)(model_input)
    x = VarDropout(300, name = model_type+'_d2',activation = tf.keras.activations.relu)(x)
    x = VarDropout(300, name = model_type+'_d3',activation = tf.keras.activations.relu)(x)

    model_out = Dense(num_labels, name = model_type+'_out')(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'dense_ref':
    print('Building dense_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data') 
    x = Dense(324, activation='relu',name = model_type+'_d1')(model_input)
    x = Dense(100, activation='relu',name = model_type+'_d2')(x)
    x = Dense(100, activation='relu',name = model_type+'_d3')(x)
    x = Dropout(.5)(x)

    model_out = Dense(num_labels, name = model_type)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model    

  elif model_type == 'pdf':
    print('Building Encoder Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    
    _1, _2, z = encoder(model_input)
    x = Concatenate()([model_input,z,_1,_2])
    x = Dense(300,  activation =  'relu',name=model_type+'_d1')(x)
    x = Dense(300,  activation =  'relu', name=model_type+'_d2')(x)
    x = Dense(300,  activation =  'relu', name=model_type+'_d3')(x)

    model_out = Dense(num_labels,name=model_type+'_output')(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'pd':
    print('Building Probability Dropout MLP Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    
    
    x = Dense(300,activation = 'relu', name=model_type+'_d1')(model_input)
    x = PGD(predictions, 300,name=model_type+'_pd1')(x)
    x = Dense(300,activation = 'relu', name=model_type+'_d2')(x)
    x = PGD(predictions, 300,name=model_type+'_pd2')(x)
    x = Dense(300,activation = 'relu', name=model_type+'_d3')(x)
    x = PGD(predictions, 300,name=model_type+'_pd3')(x)

    model_out = Dense(num_labels,name=model_type+"_output")(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'stack':
    print('Building Encoder-MLP Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _,_, z = encoder(model_input)
    z_mean = z[:,0]
    z_log_var = z[:,1]
    x = Dense(300,activation = 'relu')(z)
    x = Dense(300,activation = 'relu')(x)
    x = Dense(300,activation = 'relu')(x)
    x = Dropout(dropout_rate)(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()

    kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # add loss to model
    # model_loss = tf.reduce_mean(loss_fn + kl_loss)
    model.add_loss(kl_loss)
    return model

  elif model_type == 'sampling_dropout':
    print('Building Pre-Encoder Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(model_input)
    z_mean = z[:,0]
    z_log_var = z[:,1]
    
    x = Dense(300,activation = 'relu',name = model_type+'_d1')(model_input)
    x = VarDropout(300, name = model_type+'_vd1',activation = tf.keras.activations.relu)(x)
    x = Dense(100,activation = 'relu',name = model_type+'_d2')(x)
    x = VarDropout(100, name = model_type+'_vd2',activation = tf.keras.activations.relu)(x)
    
    model_out = Dense(num_labels, name=model_name)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    
    kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    # add loss to model
    #model_loss = tf.reduce_mean(loss_fn + kl_loss)
    model.add_loss(kl_loss)
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model
 
  elif model_type == 'conv':
    print('Building Reference CONV Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(i)
    x = Reshape(target_shape = (28, 28, 1))(i)
    x = Conv2D(filters = num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    
    model = Model(i, x, name = model_name)
    model.summary()
    plot_model(model, to_file=model_type+'.png', show_shapes=True)
    return model

  elif model_type == 'conv_stack':
    print('Building conv_stack Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(i)
    x = Sampling()
    x = keras.layers.Reshape(target_shape=(28, 28, 1))(x)
    x = keras.layers.Conv2D(filters=num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = keras.layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(10)

    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    return model
    
  else:   
    print('Building Reference Lenet324_100_100 Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)

    model_out = Dense(num_labels, name=model_type +'_out')(x)
    model = Model(model_input, model_out, name = model_name)
    model.summary()
    return model


def create_vae(x_train):
    input_dim = output_dim = x_train.shape[-1]
    
    # Define encoder model.  
    i = tf.keras.Input(shape=(input_dim,), name=_vae.model_type +"enc_input")
    x = Dense(_vae.intermediate, activation="relu",name=_vae.model_type +"_enc_d1")(i)
    z_mean = Dense(_vae.latent, name=_vae.model_type +"_z_mean")(x)
    z_log_var = Dense(_vae.latent, name=_vae.model_type +"_z_log_var")(x)

    if _vae.model_type == 'ref':
      z = Sampling(name=_vae.model_type +"_z")([z_mean, z_log_var])      
    else:
      x = Concatenate()([z_mean,z_log_var])
      z = VarDropout(_vae.latent, activation=tf.nn.relu, name=_vae.model_type +"_zvd")(x)

    enc = Model(inputs=i, outputs=[z_mean, z_log_var, z], name=_vae.enc_name)
    

    # Define decoder model.
    li = Input(shape=(_vae.latent,), name=_vae.dec_name)
    x = Dense(_vae.intermediate, activation='relu',name=_vae.model_type +"_decoder_d1")(i)
    o = Dense(input_dim, name=_vae.model_type +'decoder_output',activation='sigmoid')(x)
    dec = Model(li, o, name=_vae.dec_name)


    # Define VAE model.
    outputs = dec(enc(original_inputs)[2])
    vae = tf.keras.Model(i, o, name=_vae.model_name)
    vae.summary()

    # Add KL divergence regularization loss.
    if   _vae.loss == 'mse':
      reconstruction_loss = losses.mean_squared_error(original_inputs, outputs)
    elif _vae.loss == 'msle':
      reconstruction_loss = losses.mean_squared_logarithmic_error(original_inputs, outputs)
    elif _vae.loss == 'bce':
      reconstruction_loss = losses.binary_crossentropy(original_inputs, outputs)
    elif _vae.loss == 'cce':
      reconstruction_loss = losses.categorical_crossentropy(original_inputs, outputs)
    elif _vae.loss == 'sce':      
      reconstruction_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits = outputs, labels = original_inputs)

     # Scale the loss
    reconstruction_loss *= input_dim    
    
    if _vae.global_kld:
      kl_loss = tf.keras.losses.kld(original_inputs, outputs)
      # kl_loss = 1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var)
      # kl_loss = tf.reduce_sum(kl_loss, axis=-1)
      # kl_loss *= -0.5

    else:
      kl_loss = 0

    # Add loss to model
    vae_loss = tf.reduce_mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)

    optimizer = tf.keras.optimizers.Adam()
    vae.compile(optimizer)
        
    return vae, encoder, decoder


# Training settings
EPOCHS = 1
BATCH_SIZE = 128
VAE_EPOCHS = 10
CUSTOM_TRAIN = False


##########################################################################
# Settings
#

class VaeSettings(object):
  model_type = 'vd'
  loss = 'msle'  
  intermediate = 512
  latent = 2

  # global kld or local only (VD)
  global_kld = True
  load_weights = False

  vae_name = "vae-{}-{}-{}".format(model_type, intermediate, latent)
  enc_name = "enc-{}-{}-{}".format(model_type, intermediate, latent)
  dec_name = "dec-{}-{}-{}".format(model_type, intermediate, latent)

_vae = VaeSettings()  


class ModelSettings(object):
  model_type = 'stack'
  dataset = 'mnist'
  
  enc_trainable = True
  loss_fn = tf.losses.CategoricalCrossentropy(from_logits = True)
  optimizer = tf.keras.optimizers.Adam()
  metrics = [keras.metrics.CategoricalAccuracy()]

  model_name = "{}-{}".format(model_type,dataset)

_md = ModelSettings()
save_dir = os.path.join(os.getcwd(), 'saved_models')
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")




if __name__ == '__main__':

  # parameters

  # load data
  if _md.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_minst_data(True)
  elif _md.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_cifar10_data(True)



  # gather predictions for the test batch
  _, _, predictions = encoder.predict(x_test, batch_size=BATCH_SIZE) 
  
  initial_values = utils.get_varparams_class_means(predictions, y_test, num_labels)
  p = tf.stack(predictions)
  


  ####################################################################
  # Create
  #
 
  # vae
  vae, encoder, decoder = create_vae(x_train)



  # model

  model = create_model(
                x_train, 
                initial_values, 
                num_labels, 
                encoder, 
                _md.loss_fn ,
                _md.model_type,
                predictions = predictions,                
                dropout_rate = .2)

  model.compile(optimizer,
            loss = _md.loss_fn, 
            metrics = _md.metrics,
            experimental_run_tf_function = False)



  ####################################################################
  # Train
  #

  if CUSTOM_TRAIN:
    Custom_train(model, x_train, y_train, optimizer, x_test, y_test, loss_fn)  

  else:  
    sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch') 
    tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                  histogram_freq = 2,
                                                  write_graph=True,
                                                  write_images=True,
                                                  update_freq = 'epoch',
                                                  profile_batch = [2,10]
                                                  )

    vae.fit(x_train, x_train, 
            epochs = VAE_EPOCHS, 
            batch_size = BATCH_SIZE,
            validation_split = .1,
            shuffle = True,
            callbacks =[tensorboard_cb],
            verbose=1
          )
    
    if not _md.enc_trainable:
      encoder.trainable = False

    model.fit(x_train, y_train, 
              epochs=EPOCHS, 
              batch_size=BATCH_SIZE,            
              validation_split = .01,
              shuffle=True,
              callbacks=[ tensorboard_cb],
              verbose=1
              )



  ####################################################################
  # Plot
  #
    
  plot_model(vae, to_file=_vae.vae_name + '.png', show_shapes=True, show_layer_names=False)
  plot_model(enc, to_file=_vae.enc_name + '.png', show_shapes=True, show_layer_names=False)
  plot_model(dec, to_file=_vae.dec_name + '.png', show_shapes=True, show_layer_names=False)

  model_writer = tf.summary.create_file_writer(log_dir + '\\vae')
  model_writer.set_as_default()

  latent_fig = utils.plot_encoding(encoder,
                  [x_test, y_test],
                  batch_size=BATCH_SIZE,
                  model_name=_vae.model_type)

  reconstruct_fig = utils.plot_reconstruction(encoder,decoder,x_test)                   

  # generate hinton
  # l = model.get_layer('layer2')
  # weights = l.get_weights()
  # theta = weights[0]
  # utils.hinton(theta)

  ###################################################################################
  # Log
  #

  model_writer = tf.summary.create_file_writer(log_dir + '\\images')
  model_writer.set_as_default()
  with model_writer.as_default():
    tf.summary.image("Latent Space Plot", utils.plot_to_image(latent_fig), step = 0)
    tf.summary.image("Reconstruction Plot", utils.plot_to_image(reconstruct_fig), step = 0)

  
  ####################################################################
  # Evaluate
  #
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE)  
  print('\n')
  print(str(model_type) + ' Model Test Loss : ', score[0])
  print(str(model_type) + ' Test Accuracy : %.1f%%' % (100.0 * score[1]))

