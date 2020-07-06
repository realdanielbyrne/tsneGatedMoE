from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import argparse
from tensorflow import keras

from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.datasets import cifar10, mnist

EPSILON = 1e-8

#######################################################
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

def vd_loss(model):
  log_alphas = []
  fraction = 0.
  theta_logsigma2 = [layer.variables for layer in model.layers if 'var_dropout' in layer.name]
  for theta, log_sigma2, step in theta_logsigma2:
    log_alphas.append(compute_log_alpha(theta, log_sigma2))
    fraction = tf.minimum(tf.maximum(fraction,step),1.)
  return log_alphas, fraction

@tf.function
def compute_log_alpha(theta, log_sigma2):
  return log_sigma2 - tf.math.log(tf.square(theta) + EPSILON)

def parse_cmd(description = 'AE Embedding Classifier'):
  parser = argparse.ArgumentParser(description=description)
  parser.add_argument("-c", "--categorical",
                      default=True,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")


  parser.add_argument("-ds", "--dataset",
                      action='store',
                      type=str,
                      default='mnist',
                      help="Use sparse, integer encoding, instead of one-hot")
                  


  args = parser.parse_args()
  return args

def load_minst_data(categorical):
      # load mnist dataset
  (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
  # compute the number of labels
  num_labels = len(np.unique(y_train))

  # image dimensions (assumed square)
  image_size = x_train.shape[1]
  input_size = image_size * image_size

  # resize and normalize
  x_train = np.reshape(x_train, [-1, input_size]).astype('float32') / 255.
  x_test = np.reshape(x_test, [-1, input_size]).astype('float32') / 255.

  if categorical:
    # Convert class vectors to binary class matrices ( One Hot Encoding )
    y_train = to_categorical(y_train)
    y_test_cat = to_categorical(y_test)
  else:
    y_test_cat = y_test

  return (x_train, y_train), (x_test, y_test), num_labels, y_test_cat

def load_cifar10_data(categorical):
  # load the CIFAR10 data
  K.set_image_data_format('channels_first')
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  # compute the number of labels
  num_labels = len(np.unique(y_train))

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255.
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255.

  if categorical:
    # Convert class vectors to binary class matrices ( One Hot Encoding )
    y_train = to_categorical(y_train, 10)
    y_test_cat = to_categorical(y_test)
  else:
    y_test_cat = y_test    

  return (x_train, y_train), (x_test, y_test), num_labels, y_test_cat

def plot_encoding(encoder,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as a function of the 2D latent vector
    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test, batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()


def plot_label_clusters(encoder, data, labels):
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(data)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=labels)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.show()


def plot_layer_activations(model,x_test,y_test):
  
  from tensorflow.keras import backend as K
  model_in = model.input               # input placeholder
  model_out = [layer.output for layer in model.layers if 'dense' in layer.name] # all layer outputs
  fun = K.function([model_in, False], model_out) # evaluation function

  # Testing
  layer_outputs = fun([x_test, 1.])

def layer_to_visualize(model,layer,test_image):
    '''
    # Specify the layer to want to visualize
    layer_to_visualize(convout1)
    '''
    inputs = [K.learning_phase()] + model.inputs

    _convout1_f = K.function(inputs, [layer.output])
    def convout1_f(X):
        # The [0] is to disable the training phase flag
        return _convout1_f([0] + [X])

    convolutions = convout1_f(test_image)
    convolutions = np.squeeze(convolutions)

    print ('Shape of conv:', convolutions.shape)

    n = convolutions.shape[0]
    n = int(np.ceil(np.sqrt(n)))

    # Visualization of each filter of the layer
    fig = plt.figure(figsize=(12,8))
    for i in range(len(convolutions)):
        ax = fig.add_subplot(n,n,i+1)
        ax.imshow(convolutions[i], cmap='gray')

  
#function to get activations of a layer
def get_activations(model, layer, x_train):
    get_activations = K.function([model.layers[0].input, K.learning_phase()], [model.layers[layer].output,])
    activations = get_activations([x_train,0])
    return activations

#Get activations using layername
def get_activation_from_layer(model,layer_name,layers,layers_dim,img):
  acti = get_activations(model, layers[layer_name], img.reshape(1,256,256,3))[0].reshape(layers_dim[layer][0],layers_dim[layer_name][1],layers_dim[layer_name][2])
  return np.sum(acti,axis=2)  

# #Map layer name with layer index
# layers = dict()
# index = None
# for idx, layer in enumerate(model.layers):
#   layers[layer.name] = idx

# #Map layer name with its dimension
# layers_dim = dict()

# for layer in model.layers:
#   layers_dim[layer.name] = layer.get_output_at(0).get_shape().as_list()[1:]

# img1 = utils.load_img("image.png", target_size=(256, 256))

# #define the layer you want to visualize
# layer_name = "conv2d_22"
# plt.imshow(get_activation_from_layer(model,layer_name,layers,layers_dim, img1), cmap="jet")


def print_names_and_shapes(activations: dict):
    for layer_name, layer_activations in activations.items():
        print(layer_name, layer_activations.shape)
    print('-' * 80)


def print_names_and_values(activations: dict):
    for layer_name, layer_activations in activations.items():
        print(layer_name)
        print(layer_activations)
        print('')
    print('-' * 80)


def gpu_dynamic_mem_growth():
    # Check for GPUs and set them to dynamically grow memory as needed
    # Avoids OOM from tensorflow greedily allocating GPU memory
    try:
        gpu_devices = tf.config.list_physical_devices('GPU')
        if len(gpu_devices) > 0:
            for gpu in gpu_devices:
                tf.config.experimental.set_memory_growth(gpu, True)
    except AttributeError:
        print('Upgrade your tensorflow to 2.x to have the gpu_dynamic_mem_growth feature.')