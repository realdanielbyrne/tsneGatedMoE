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

from tensorflow.keras.utils import plot_model
import matplotlib.pyplot as plt
from layers import VarDropout
import utils
import keract

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
        logits, z_mean, z_log_var = inputs        
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
                model_type = 'cgd',
                dropout_rate = .2):
  

  if model_type == 'cgd':
    print('Building CGD Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    x = Dense(300)(model_input)
    y = []
    for i in range(num_labels):
      #y.append(CGDDropoutLeNetBlock(initial_values[i], activation = tf.nn.sigmoid)(x))      
      y.append(ConstantGausianDropoutGate(initial_values[i], num_outputs = 100,activation = tf.nn.sigmoid)(x))
    x = Concatenate()(y)
    x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001),name='dense2')(x)
    x = Dense(100,kernel_regularizer=tf.keras.regularizers.l2(0.001),name='dense3')(x)
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model


  elif model_type == 'preencoder':
    print('Building Pre-Encoder Model')
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    z, zm, zlv = encoder(model_input)
    z_mean = z[:,0]
    z_log_var = z[:,1]
    x = Dense(300)(model_input)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100)(x)
    x = SamplingDropout()([x,z[:,0],z[:,1]])
    x = Dense(100)(x)
    
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model
  
  elif model_type == 'vae':   
    print('Building VAE Filter Model') 
    model_input = keras.layers.Input(shape = (x_train.shape[-1],), name='data')
    vae_out = decoder(encoder(model_input)[2])    
    x = Dense(300)(model_input)
    x = Dense(100)(x)
    x = Dense(100)(x)
    x = Dropout(.2)(x)
    model.summary()
    return model

  elif model_type == 'conv':
    print('Building Reference CONV Model')
    model = keras.Sequential([
      keras.layers.InputLayer(input_shape=(28, 28)),
      keras.layers.Reshape(target_shape=(28, 28, 1)),
      keras.layers.Conv2D(filters=num_labels, kernel_size=(3, 3), activation='relu',),
      keras.layers.MaxPooling2D(pool_size=(2, 2)),
      keras.layers.Flatten(),
      keras.layers.Dense(10)
    ])
    model_out = Dense(num_labels, name=model_type)(x)
    model = Model(model_input, model_out, name = model_type)
    model.summary()
    return model

  elif model_type == 'conv_cgd':
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
  x_val = x_train[-3000:]
  y_val = y_train[-3000:]
  x_train = x_train[:-3000]
  y_train = y_train[:-3000]
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

      # # Add kld layer losses created during this forward pass:
      # log_alphas,dkl_fraction = vd_loss(model)
      # dkl_loss = tf.add_n([negative_dkl(log_alpha=a) for a in log_alphas])

      # regularizer intensifies over the course of ramp-up
      # tf.summary.scalar('dkl_fraction', dkl_fraction)
      # tf.summary.scalar('dkl_loss_gross',dkl_loss )
      # dkl_loss = dkl_loss * dkl_fraction

      
      dkl_loss =  sum(model.losses) 
      tf.summary.scalar('dkl_loss_net',dkl_loss)
      #dkl_loss = dkl_loss / float(x_train.shape[0])
      loss_value += dkl_loss
      
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


def negative_dkl(log_alpha=None):
  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1
  # Compute each term of the KL and combine
  term_1 = k1 * tf.nn.sigmoid(k2 + k3*log_alpha)
  term_2 = -0.5 * tf.math.log1p(tf.math.exp(tf.math.negative(log_alpha)))
  eltwise_dkl = term_1 + term_2 + c
  return -tf.reduce_sum(eltwise_dkl)



# Settings
original_dim = 784
EPOCHS = 10
intermediate_dim = 512
BATCH_SIZE = 128
latent_dim = 2
vae_epochs = 10
override = False

if __name__ == '__main__':

  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-c", "--categorical",
                      default=True,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")
  parser.add_argument("-s", "--embedding_type",
                      default='mean',
                      help="embedding_type - sample: Samples a single x_test latent variable for each class\n\
                            mean: Averages all x_test latent variables")

  parser.add_argument("-m", "--model_type",
                      default='cgd',
                      help="model_type - sample: Model under test.  vae, cgd, preencoder")

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
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]
  


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

  

  # gather predictions for the test batch
  predictions, _, _ = encoder.predict(x_test, batch_size=BATCH_SIZE) 
  initial_values = utils.get_varparams_class_means(predictions, y_test, num_labels)

  # create model under test
  model = create_model(x_train, initial_values, num_labels, encoder, model_type='cgd')
  
  loss_fn = tf.losses.CategoricalCrossentropy(from_logits = True)
  metrics = [keras.metrics.CategoricalAccuracy()]

  STEPS_PER_EPOCH = x_train.shape[0]//BATCH_SIZE
  lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    0.001,
    decay_steps=STEPS_PER_EPOCH*100,
    decay_rate=1,
    staircase=True)

  optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
  # Train
  # use custom training loop to assist in debugging
  #custom_train(model, x_train, y_train, optimizer, x_test, y_test, loss_fn)
  
  # use graph training for speed
  model.compile(optimizer,loss = loss_fn, metrics=['accuracy'],experimental_run_tf_function=False)
  model.fit(x_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE,callbacks=[tensorboard_cb], validation_split=.01)
  plot_model(model,to_file= 'plots\\mlp_vd.png')


  #model accuracy on test dataset
  score = model.evaluate(x = x_test, y = y_test_cat, batch_size=BATCH_SIZE)
  print('\nMLP Control Model Test Loss:', score[0])
  print("MLP Control Model Test Accuracy: %.1f%%" % (100.0 * score[1]))

  def plot_layer_activations(model,x_test,y_test):

    from tensorflow.keras import backend as K
    model_in = model.input               # input placeholder
    model_out = [layer.output for layer in model.layers if 'dense' in layer.name] # all layer outputs
    fun = K.function([model_in], model_out) # evaluation function
    
    layer_outputs = fun([x_test[:100], 1.])   

    x = np.ones((layer_outputs[1].shape))*np.expand_dims(y_test[:100],1)
    x = x*tf.random.normal(shape=(layer_outputs[1].shape),mean=1,stddev=.01)
    y = np.ones((layer_outputs[1].shape))*y
    c = tf.nn.softmax(np.squeeze(layer_outputs[1]))*100    
    plt.scatter(x,y,c=c,cmap='Blues',alpha = .5)
    plt.colorbar()
    plt.show()

    x = np.ones((layer_outputs[2].shape))*np.expand_dims(y_test[:100],1)
    x = x*tf.random.normal(shape=(layer_outputs[2].shape),mean=1,stddev=.01)
    y = np.ones((layer_outputs[2].shape))*y
    c = tf.nn.softmax(np.squeeze(layer_outputs[2])) *100   
    plt.scatter(x,y,c=c,cmap='Blues',alpha = .5)
    plt.colorbar()
    plt.show()


  plot_layer_activations(model,x_test,y_test)