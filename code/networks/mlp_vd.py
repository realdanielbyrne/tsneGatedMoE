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
from tensorflow.keras.layers import Concatenate, Softmax, Conv2D, MaxPooling2D, Flatten, Layer, Multiply, Add, Subtract, Average, Activation
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
    model = Model(model_input, model_out, name = _md.model_name)
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
    model = Model(model_input, model_out, name = _md.model_name)
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
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'vd_ref':
    print('Building vd_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name=model_type + '_in') 
    x = VarDropout(300, name = model_type+'_d1',activation = tf.keras.activations.relu)(model_input)
    x = VarDropout(300, name = model_type+'_d2',activation = tf.keras.activations.relu)(x)
    x = VarDropout(300, name = model_type+'_d3',activation = tf.keras.activations.relu)(x)

    model_out = Dense(num_labels, name = model_type+'_out')(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'dense_ref':
    print('Building dense_ref Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data') 
    x = Dense(324, activation='relu',name = model_type+'_d1')(model_input)
    x = Dense(100, activation='relu',name = model_type+'_d2')(x)
    x = Dense(100, activation='relu',name = model_type+'_d3')(x)
    x = Dropout(.5)(x)

    model_out = Dense(num_labels, name = model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
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
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'mlp-stack':
    print('Building Encoder-MLP Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _,_, z = encoder(model_input)

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

  elif model_type == 'conv-stack':
    print('Building Encoder-CONV Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _,_, z = encoder(model_input)

    model = Sequential (
      Conv2D(input_shape=x_train[0,:,:,:].shape, filters=96, kernel_size=(3,3)),
      Activation('relu'),
      Conv2D(filters=96, kernel_size=(3,3), strides=2),
      Activation('relu'),
      Dropout(0.2),
      Conv2D(filters=192, kernel_size=(3,3)),
      Activation('relu'),
      Conv2D(filters=192, kernel_size=(3,3), strides=2),
      Activation('relu'),
      Dropout(0.5),
      Flatten(),
      BatchNormalization(),
      Dense(256),
      Activation('relu'),
      Dense(num_labels, activation="softmax")
    )
    model.name = _md.model_name

    if _md.kld_loss:
      kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
      kl_loss = tf.reduce_sum(kl_loss, axis=-1)
      kl_loss *= -0.5
      model.add_loss(kl_loss)
    
    return model

  elif model_type == 'smconv_ref':
    print('Building Reference small conv Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    if _md.dataset == 'mnist':
      x = Reshape(target_shape = (28, 28, 1))(i)
    x = Conv2D(filters = num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    
    model = Model(i, x, name = _md.model_name)
    return model

  elif model_type == 'smconv_stack':
    print('Building small conv_stack Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(i)


    x = Reshape(target_shape = (28, 28, 1))(x)
    x = Flatten()(x)
    
    x = Conv2D(filters = num_labels, kernel_size = (3, 3), activation = 'relu')(x)
    x = MaxPooling2D(pool_size = (2, 2))(x)
    x = Flatten()(x)
    x = Dense(10)

    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    return model

  elif model_type == 'sampling_dropout':
    print('Buildingsampling_dropout Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')

    x = Dense(300,activation = 'relu',name = model_type+'_d1')(model_input)
    x = VarDropout(300, name = model_type+'_vd1',activation = tf.keras.activations.relu)(x)
    x = Dense(100,activation = 'relu',name = model_type+'_d2')(x)
    x = VarDropout(100, name = model_type+'_vd2',activation = tf.keras.activations.relu)(x)
    
    model_out = Dense(num_labels, name=model_name)(x)
    model = Model(model_input, model_out, name = _md.model_name)
    model.summary()
    
    kl_loss = 1 + z_log_var - tf.math.square(z_mean) - tf.math.exp(z_log_var)
    kl_loss = tf.reduce_sum(kl_loss, axis=-1)
    kl_loss *= -0.5

    #add loss to model
    model_loss = tf.reduce_mean(loss_fn + kl_loss)
    model.add_loss(kl_loss)
    return model
 
  elif model_type == 'conv':
    print('Building Reference CONV Model')
    model = Sequential (
      Conv2D(input_shape=x_train[0,:,:,:].shape, filters=96, kernel_size=(3,3)),
      Activation('relu'),
      Conv2D(filters=96, kernel_size=(3,3), strides=2),
      Activation('relu'),
      Dropout(0.2),
      Conv2D(filters=192, kernel_size=(3,3)),
      Activation('relu'),
      Conv2D(filters=192, kernel_size=(3,3), strides=2),
      Activation('relu'),
      Dropout(0.5),
      Flatten(),
      BatchNormalization(),
      Dense(256),
      Activation('relu'),
      Dense(num_labels, activation="softmax")
    )
    model.name = _md.model_name
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
EPOCHS = 10
BATCH_SIZE = 128
VAE_EPOCHS = 20
CUSTOM_TRAIN = False
BUILD_VAE = False

##########################################################################
# Settings
#

class VaeSettings(object):
  model_type = 'ref'
  global_kld = True

  loss = 'msle'  
  intermediate = 512
  latent = 2
  optimizer = tf.keras.optimizers.Adam()
  act = 'relu'

  if global_kld:
    reg = 'g'
  else:
    reg = 'l'

  vae_name = "vae-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act)
  enc_name = "enc-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act)
  dec_name = "dec-{}-{}-{}-{}-{}-{}".format(model_type, intermediate, latent, loss, reg, act)
_vae = VaeSettings()  

def myLoss(y_true,y_pred):
  return tf.nn.softmax_cross_entropy_with_logits(y_true,y_pred)

class ModelSettings(object):
  model_type = 'smconv_ref'
  dataset = 'mnist'
  
  kld_loss = True
  enc_trainable = True
  loss_fn = myLoss
  #loss_fn = tf.keras.losses.CategoricalCrossentropy(from_logits = True)
  optimizer = tf.keras.optimizers.Adam()
  metrics = [keras.metrics.CategoricalAccuracy()]

  if enc_trainable:
    t = 'trn'
  else:
    t = 'lock'
  
  if kld_loss:
    reg = 'kld'
  else:
    reg = ''
  
  model_name = "{}-{}-{}-{}".format(model_type, dataset, t, reg)
_md = ModelSettings()

save_dir = os.path.join(os.getcwd(), 'saved_models')
log_dir = "logs\\fit\\" + datetime.now().strftime("%Y%m%d-%H%M%S")


if __name__ == '__main__':

  # load data
  if _md.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_minst_data(True)
  elif _md.dataset == 'cifar10':
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_cifar10_data(True)


  ####################################################################
  # vae
  #

  # Callbacks
  sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch') 
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, 
                                                histogram_freq = 2,
                                                write_graph=True,
                                                write_images=True,
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
    plot_model(vae, to_file= "plots\\" +_vae.vae_name + '.png', show_shapes=True, show_layer_names=False)
    plot_model(enc, to_file= "plots\\" +_vae.enc_name + '.png', show_shapes=True, show_layer_names=False)
    plot_model(dec, to_file= "plots\\" +_vae.dec_name + '.png', show_shapes=True, show_layer_names=False)

    latent_fig = utils.plot_encoding(enc,
                    [x_test, y_test],
                    batch_size=BATCH_SIZE,
                    model_name=_vae.model_type)

    utils.plot_reconstruction(dec, x_test)                   

  else:
    vae = tf.keras.models.load_model('models/'+ _vae.vae_name)
    enc = tf.keras.models.load_model('models/'+ _vae.enc_name)
    dec = tf.keras.models.load_model('models/'+ _vae.dec_name)
    vae.compile

  
  # gather predictions for the test batch
  _, _, predictions = enc.predict(x_test, batch_size=BATCH_SIZE) 
  
  # calculate class means
  initial_values = utils.get_varparams_class_means(predictions, y_test, num_labels)

  

  ####################################################################
  # model
  #
  model = create_model(
                x_train, 
                initial_values, 
                num_labels, 
                enc, 
                _md.loss_fn ,
                _md.model_type,
                predictions = predictions,                
                dropout_rate = .2)
  model.summary()

  model.compile(_md.optimizer,
            loss = _md.loss_fn, 
            metrics = _md.metrics,
            experimental_run_tf_function = False)

  model.summary()
  plot_model(model, to_file = "plots/"+_md.model_name +'.png', show_shapes = True)
  


  ####################################################################
  # Train
  #
  if CUSTOM_TRAIN:
    Custom_train(model, x_train, y_train, _md.optimizer, x_test, y_test, loss_fn)  

  else:  
      
    if not _md.enc_trainable:
      encoder.trainable = False

    model.fit(x_train, y_train, 
              epochs = EPOCHS, 
              batch_size = BATCH_SIZE,            
              validation_split = .01,
              shuffle = True,
              callbacks=[tensorboard_cb],
              verbose=1
              )
  
  ####################################################################
  # Evaluate
  #
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE)  
  print('\n')
  print(str(_md.model_name) + ' Model Test Loss : ', score[0])
  print(str(_md.model_name) + ' Test Accuracy : %.1f%%' % (100.0 * score[1]))

