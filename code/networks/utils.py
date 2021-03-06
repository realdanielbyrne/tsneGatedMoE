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
from tensorflow.keras.datasets import cifar10, mnist, fashion_mnist
from sklearn.decomposition import PCA
EPSILON = 1e-8



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

def get_classp(predictions, y_test, num_labels):

  classpreds = []
  for x in range(num_labels):
    targets = predictions[np.where(y_test == x)[0]]
    classpreds.append(targets)
    
  return classpreds

def get_varparams_class_params(predictions, y_test, num_labels):
  initial_thetas = []
  initial_log_sigma2s = []
  
  for x in range(num_labels):
    targets = predictions[np.where(y_test == x)[0]]
    targets = tf.stack(targets)

    mean, var = tf.nn.moments(targets, axes = 0)
    rmean = np.mean(mean)
    rvar = np.mean(var)
    initial_thetas.append(rmean)
    initial_log_sigma2s.append(rvar )

  initial_values = np.transpose(np.stack([initial_thetas,initial_log_sigma2s]))
  return initial_values


def load_dataset(dataset, categorical):
  if dataset == 'mnist' or dataset == 'fashion_mnist' :
    if dataset == 'mnist':
      (x_train, y_train), (x_test, y_test) = mnist.load_data()
    elif dataset == 'fashion_mnist':
      (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

    # image dimensions (assumed square)
    image_size = x_train.shape[1]
    input_size = image_size * image_size

    # resize and normalize
    x_train = np.reshape(x_train, [-1, input_size]).astype('float32') / 255.
    x_test = np.reshape(x_test, [-1, input_size]).astype('float32') / 255.
    
  elif dataset == 'cifar10':    
    K.set_image_data_format('channels_last')
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    n, channel, row, col = x_train.shape
    x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255.
    x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255.

  # compute the number of labels
  num_labels = len(np.unique(y_train))

  if categorical:
    # Convert class vectors to binary class matrices ( One Hot Encoding )
    y_train = to_categorical(y_train, num_labels)
    y_test_cat = to_categorical(y_test)
  else:
    y_test_cat = y_test    

  return (x_train, y_train), (x_test, y_test), num_labels, y_test_cat

def plot_to_image(figure):
  import io
  """Converts the matplotlib plot specified by 'figure' to a PNG image and
  returns it. The supplied figure is closed and inaccessible after this call."""
  # Save the plot to a PNG in memory.
  buf = io.BytesIO()
  plt.savefig(buf, format='png')
  # Closing the figure prevents it from being displayed directly inside
  # the notebook.
  plt.close(figure)
  buf.seek(0)
  # Convert PNG buffer to TF image
  image = tf.image.decode_png(buf.getvalue(), channels=4)
  # Add the batch dimension
  image = tf.expand_dims(image, 0)
  return image

def plot_layer_activations(model,x_test,y_test):
  
  from tensorflow.keras import backend as K
  model_in = model.input               # input placeholder
  model_out = [layer.output for layer in model.layers if 'dense' in layer.name] # all layer outputs
  fun = K.function([model_in, False], model_out) # evaluation function

  # Testing
  layer_outputs = fun([x_test, 1.])

def sparseness(log_alphas, thresh=3):
    N_active, N_total = 0., 0.
    for la in log_alphas:
        m = tf.cast(tf.less(la, thresh), tf.float32)
        n_active = tf.reduce_sum(m)
        n_total = tf.cast(tf.reduce_prod(tf.shape(m)), tf.float32)
        N_active += n_active
        N_total += n_total
    return 1.0 - N_active/N_total


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


def plot_layer_activations(model,x_test,y_test, layername):

  from tensorflow.keras import backend as K
  model_in = model.input               # input placeholder
  model_out = [layer.output for layer in model.layers if layername in layer.name] # all layer outputs
  fun = K.function([model_in], model_out) # evaluation function
  
  layer_outputs = fun([x_test, 1.])   

  x = np.ones((layer_outputs[1].shape)) * np.expand_dims(y_test[:100],1)
  x = x*tf.random.normal(shape=(layer_outputs[1].shape),mean=1,stddev=.01)
  y = range(100)
  y = np.ones((layer_outputs[1].shape))*y
  c = tf.nn.softmax(np.squeeze(layer_outputs[1]))*100    
  plt.scatter(x, y, c=c, cmap='Blues', alpha = .5)
  plt.colorbar()
  plt.xlabel("Label Class")
  plt.ylabel("Layer Neuron")
  plt.savefig('dense_layer_activations_by_class2.png')
  plt.show()


def hinton(matrix, max_weight=None, ax=None):
    """Draw Hinton diagram for visualizing a weight matrix.
    
    
    # generate hinton
    # l = model.get_layer('layer2')
    # weights = l.get_weights()
    # theta = weights[0]
    # utils.hinton(theta)

    """
    ax = ax if ax is not None else plt.gca()

    if not max_weight:
        max_weight = 2 ** np.ceil(np.log(np.abs(matrix).max()) / np.log(2))

    ax.patch.set_facecolor('gray')
    ax.set_aspect('equal', 'box')
    ax.xaxis.set_major_locator(plt.NullLocator())
    ax.yaxis.set_major_locator(plt.NullLocator())

    for (x, y), w in np.ndenumerate(matrix):
        color = 'white' if w > 0 else 'black'
        size = np.sqrt(np.abs(w) / max_weight)
        rect = plt.Rectangle([x - size / 2, y - size / 2], size, size,
                             facecolor=color, edgecolor=color)
        ax.add_patch(rect)

    ax.autoscale_view()
    ax.invert_yaxis()

def plot_layer_weights(model,layername = 'layer2'):
  l = model.get_layer(layername)
  weights = l.get_weights()
  
  fig, axes = plt.subplots(4, 10)
  vmin, vmax = weights[0].min(), weights[0].max()
  
  for coef, ax in zip(weights[0].T, axes.ravel()):
    ax.matshow(coef.reshape(28, 28), cmap=plt.cm.gray, vmin=.5 * vmin,
              vmax=.5 * vmax)
    ax.set_xticks(())
    ax.set_yticks(())
  
  plt.show()


def plot_mlp_activations(model, x_test, y_test, num_labels = 10):
  
  from tensorflow.keras import backend as K
  model_in = model.input               # input placeholder
  model_out = [layer.output for layer in model.layers if 'layer1' in layer.name] # all layer outputs
  fun = K.function([model_in], model_out) # evaluation function
  
  layer_outputs = fun([x_test, 1.])  
  z = layer_outputs[0]

  results = []
  x = 0
  for x in range(num_labels):
    targets = z[np.where(y_test == x)[0]]
    targets = tf.stack(targets)
    means = np.mean(targets, axis = 0)    
    results.append(means)
  
  results = np.stack(results)    
  vmin = results.min()
  vmax = results.max()

  fig, axes = plt.subplots(1, 10)
  for i,ax in zip(range(10),axes.ravel()):
    ax.matshow(results[i].reshape(18,18), cmap='Blues', vmin = .5*vmin,vmax = .5*vmax)
    ax.set_xticks(())
    ax.set_yticks(())
  
  plt.show()


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
    _, _, z = encoder.predict(x_test, batch_size=batch_size)

    if tf.shape(z)[-1] > 2:
      print('plotting first 2 principal components')
      pca = PCA(n_components = 2)
      z = pca.fit_transform(z)

    figure = plt.figure(figsize=(10, 10))
    plt.scatter(z[:, 0], z[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1")
    plt.xlim(-4,4)
    plt.ylim(-4,4)
    plt.savefig("plots/" + encoder.name )
    plt.show()


def plot_reconstruction(decoder, x_test):
  n = 30
  digit_size = 28
  figure = np.zeros((digit_size * n, digit_size * n))
  # linearly spaced coordinates corresponding to the 2D plot
  # of digit classes in the latent space
  grid_x = np.linspace(-4, 4, n)
  grid_y = np.linspace(-4, 4, n)[::-1]

  for i, yi in enumerate(grid_y):
      for j, xi in enumerate(grid_x):
          z_sample = np.array([[xi, yi]])
          x_decoded = decoder.predict(z_sample)
          digit = x_decoded[0].reshape(digit_size, digit_size)
          figure[i * digit_size: (i + 1) * digit_size,
                j * digit_size: (j + 1) * digit_size] = digit

  plt.figure(figsize=(10, 10))
  start_range = digit_size // 2
  end_range = n * digit_size + start_range + 1
  pixel_range = np.arange(start_range, end_range, digit_size)
  sample_range_x = np.round(grid_x, 1)
  sample_range_y = np.round(grid_y, 1)

  plt.xticks(pixel_range, sample_range_x)
  plt.yticks(pixel_range, sample_range_y)
  plt.xlabel("z[0]")
  plt.ylabel("z[1]")
  plt.imshow(figure, cmap='Greys_r')
  plt.savefig("plots/" + decoder.name)  
  plt.show()
  
