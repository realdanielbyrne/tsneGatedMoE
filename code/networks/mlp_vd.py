from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse

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

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from layers import VarDropout,ProbabilityDropout

import utils
import keract

EPSILON = 1e-8

class Pdf(tf.keras.initializers.Initializer):

  def __init__(self, mean, stddev):
    self.mean = mean
    self.stddev = stddev

  def __call__(self, shape, dtype=None):
    return tf.nn.softmax(tf.random.normal(
        shape, mean=self.mean, stddev=self.stddev, dtype=dtype))

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
    initial_theta, initial_log_sigma2 = initial_values

    # define static lookups for pre-calculated datasets
    self.initial_theta = initial_theta
    self.initial_log_sigma2 = initial_log_sigma2

  def build(self, input_shape):
    if self.num_outputs is None:
      self.num_outputs = input_shape[-1]

    kernel_shape = (input_shape[-1],self.num_outputs)
    
    self.kernel = self.add_weight(shape = kernel_shape,
                            #initializer= Pdf(self.initial_theta,self.initial_log_sigma2),
                            initializer=tf.keras.initializers.RandomNormal(self.initial_theta,self.initial_log_sigma2),
                            regularizer=tf.keras.regularizers.L1L2(0, 0.001),
                            trainable=True)
                            
    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = tf.keras.initializers.Constant(self.initial_theta),
                            trainable = False)
    else:
      self.b = None

  def call(self, inputs, training = None):
    val = tf.matmul(inputs, self.kernel)     

    if self.use_bias:
      val = tf.nn.bias_add(val, self.bias)

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
    pfilt = tf.sigmoid(pfilt)
    pfilt2 = tf.transpose(tf.reshape(pfilt,(-1,1)))
    filt = tf.repeat(pfilt2,repeats =input_shape[-1], axis = 0)

    self.filter = tf.Variable(filt, trainable = True)

  def call(self, inputs):
    return tf.matmul(inputs,self.filter)

class DropoutLeNetBlock(Layer):
  def __init__(self, activation = None, rate = .2):
      super(DropoutLeNetBlock, self).__init__()
      self.activation = activation
      self.dense_1 = Dense(300, activation = 'relu')
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

class SamplingDropout(Layer):
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
      epsilon = tf.random.normal(shape=tf.shape(z_mean))
      return tf.keras.activations.elu(z_mean + tf.math.exp(0.5 * z_log_var) * epsilon)

def create_model(
                x_train, 
                initial_values, 
                num_labels, 
                encoder, 
                predictions = None,
                model_type = 'stack',
                dropout_rate = .2):
  
  print('\n\n') 
  if model_type == 'cgd':
    print('Building CGD Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data') 
    x = Dense(300)(model_input)   
    
    y = []
    for i in range(num_labels):  
      y.append(ConstantGausianDropoutGate(initial_values[i], num_outputs = 300, activation = tf.nn.relu)(model_input))

    x = Concatenate()(y)
    x = Dense(200,  activation = tf.nn.relu, name='dense2')(x)
    x = Dense(100,  activation = tf.nn.relu, name='dense3')(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

  elif model_type == 'pdf':
    print('Building Encoder-Softmax Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _1, _2, x = encoder(model_input)
    #x = Concatenate()([_1,_2,x])
    x = Softmax()(x)
    x = Dense(300,  activation = tf.nn.relu, name='dense1')(x)
    x = Dense(100,  activation = tf.nn.relu, name='dense2')(x)
    x = Dense(100,  activation = tf.nn.relu, name='dense3')(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

  elif model_type == 'pd':
    print('Building Probability Dropout MLP Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    
    x = Dense(300,activation = tf.nn.relu, name='dense1')(model_input)
    x = PGD(predictions, 300)(x)
    x = Dense(100,activation = tf.nn.relu, name='dense2')(x)
    x = PGD(predictions, 100)(x)
    x = Dense(100,activation = tf.nn.relu, name='dense3')(x)
    x = PGD(predictions, 100)(x)
    d = Dropout(dropout_rate)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

  elif model_type == 'stack':
    print('Building Encoder-MLP Stack')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(model_input)
    x = Dense(300,activation = 'relu',activity_regularizer=tf.keras.regularizers.l2(0.01))(z)
    x = Dense(100,activation = 'relu',activity_regularizer=tf.keras.regularizers.l2(0.01))(x)
    #x = Add()([x,z])
    x = Dense(100,activation = 'relu',activity_regularizer=tf.keras.regularizers.l2(0.01))(x)
    #x = Add()([x,z])
    x = Dropout(.2)(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

  elif model_type == 'preencoder':
    print('Building Pre-Encoder Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(model_input)
    z_mean = z[:,0]
    z_log_var = z[:,1]
    x = Dense(300,activation = 'relu')(model_input)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100,activation = 'relu')(x)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100,activation = 'relu')(x)
    
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model
  
  elif model_type == 'vae':   
    print('Building VAE Filter Model') 
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = decoder(encoder(i)[2])    
    x = Dense(300,activation = 'relu')(x)
    x = Dense(100,activation = 'relu')(x)
    x = Dense(100,activation = 'relu')(x)
    x = Dropout(.2)(x)
    model = Model(i, x, name = model_type)
    model.summary()
    return model

  elif model_type == 'conv':
    print('Building Reference CONV Model')
    i = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    _, _, z = encoder(i)
    x = Reshape(target_shape=(28, 28, 1))(i)
    x = Conv2D(filters=num_labels, kernel_size=(3, 3), activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = Flatten()(x)
    x = Dense(10)(x)
    
    model = Model(i, x, name = model_type)
    model.summary()
    return model

  elif model_type == 'conv_stack':
    print('Building CONV_CGD Model')
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=num_labels, kernel_size=(3, 3), activation='relu'),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model
    
  else:   
    print('Building Reference Lenet300_100_100 Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = DropoutLeNetBlock(rate = dropout_rate)(model_input)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

# Settings

EPOCHS = 100
intermediate_dim = 512
BATCH_SIZE = 128
latent_dim = 9
vae_epochs = 20
override = False
model_type = 'pd'

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Probability Transfer Learning Model Builder')
  parser.add_argument("-c", "--categorical",
                      default=True,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")

  parser.add_argument("-s", "--embedding_type",
                      default='mean',
                      help="embedding_type - sample: Samples a single x_test latent variable for each class\n\
                            mean: Averages all x_test latent variables")

  parser.add_argument("-m", "--model_type",
                      default='vae',
                      help="model_type - sample: Model under test. vae, cgd, preencoder")

  parser.add_argument("-ds", "--dataset",
                      action='store',
                      type=str,
                      default='mnist',
                      help="Use sparse, integer encoding, instead of one-hot")

  args = parser.parse_args()

  # parameters
  save_dir = os.path.join(os.getcwd(), 'saved_models')
  log_dir = "logs\\fit\\" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  tensorboard_cb = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
  #sparsity_cb = tfmot.sparsity.keras.PruningSummaries(log_dir = log_dir, update_freq='epoch')
  

  # Create a callback that saves the model's weights
  checkpoint_path = "training\\cp.ckpt"
  checkpoint_dir = os.path.dirname(checkpoint_path)
  cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                  save_weights_only=True,
                                                  verbose=0)

  # load data
  if args.dataset == 'mnist':
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_minst_data(True)
  else:
    (x_train, y_train), (x_test, y_test), num_labels, y_test_cat = utils.load_cifar10_data(True)

  input_dim = output_dim = x_train.shape[-1]
  original_dim = x_test.shape[-1]
  
  if override or not os.path.isfile('models\\vae\\saved_model.pb'):
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
    vae.fit(x_train, x_train, epochs=vae_epochs, batch_size=BATCH_SIZE)

    utils.plot_encoding(encoder,
                  [x_test, y_test],
                  batch_size=BATCH_SIZE,
                  model_name="vae_mlp")
    encoder.save('models\\encoder')
    decoder.save('models\\decoder')
    vae.save('models\\vae')
  else:
    vae = tf.keras.models.load_model('models\\vae')
    encoder = tf.keras.models.load_model('models\\encoder')
    decoder = tf.keras.models.load_model('models\\decoder')
  
  #encoder.trainable = False

  # gather predictions for the test batch
  _, _, predictions = encoder.predict(x_test, batch_size=BATCH_SIZE) 
  
  initial_values = utils.get_varparams_class_means(predictions, y_test, num_labels)
  p = tf.stack(predictions)

  # create model under test
  model = create_model(x_train, initial_values, num_labels, encoder, predictions, model_type )
  
  
  loss_fn = tf.losses.CategoricalCrossentropy(from_logits = True)
  metrics = [keras.metrics.CategoricalAccuracy()]

  STEPS_PER_EPOCH = x_train.shape[0]//BATCH_SIZE
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*100,
    decay_rate=1,
    staircase=False)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  # Train
  # use custom training loop to assist in debugging
  #custom_train(model, x_train, y_train, optimizer, x_test, y_test, loss_fn)
  
  # use graph training for speed
  model.compile(optimizer,loss = loss_fn, metrics=['accuracy'],experimental_run_tf_function=False)
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_cb], validation_split=.01)
  plot_model(model,to_file= 'plots\\'+str(args.model_type)+'.png')

  #model accuracy on test dataset
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE)
  print(str(args.model_type) + '\nModel Test Loss:', score[0])
  print(str(args.model_type) + "Test Accuracy: %.1f%%" % (100.0 * score[1]))
  
  #utils.plot_layer_activations(model,x_test,y_test)