from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__doc__= """ Example usage of parametric_tSNE.
Generate some simple data in high (14) dimension, train a model,
and run additional generated data through the trained model"""

import sys
import datetime
import os

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from matplotlib.backends.backend_pdf import PdfPages


# load keras
# used plaidml so I can run on any machine's video card regardless if it is NVIDIA, AMD or Intel.
import plaidml.keras
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.utils import to_categorical

# load ptsne module
from parametric_tSNE import Parametric_tSNE
from parametric_tSNE.utils import get_multiscale_perplexities

# Globals
num_classes = 10
cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)

num_clusters = 10
test_data_tag = 'none'

def load_cifar10_data():
    # load the CIFAR10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize data
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255

    # Convert class vectors to binary class matrices.
    y_train = to_categorical(y_train, num_classes)
    y_test = to_categorical(y_test, num_classes)

    return (x_train,x_test,y_train,y_test)

def main():
  (x_train,x_test,y_train,y_test) = load_cifar10_data()
  in_dimension = x_train[1:]
  color_palette = sns.color_palette("hls", num_clusters)
  model_path_template = 'ptsne_{model_tag}_{test_data_tag}.h5'
  figure_template = 'ptsne_viz_tSNE_{test_data_tag}.pdf'
  override = True

  do_pretrain = True
  epochs = 20
  batches_per_epoch = 8
  batch_size = 128
  color_palette = sns.color_palette("hls", num_clusters)

  num_outputs = 2
  alpha_ = num_outputs - 1.0

  transformer_list = [{'label': 'Multiscale tSNE', 'tag': 'tSNE_multiscale', 'perplexity': None, 'transformer': None}]


  from tensorflow.contrib.keras import layers
  all_layers = [
    layers.Dense(500, input_shape=(x_train.shape[1:],), activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dense(500, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dense(1000, activation='relu', kernel_initializer='glorot_uniform'),
    layers.Dense(num_outputs, activation='relu', kernel_initializer='glorot_uniform')]


  for tlist in transformer_list:
    perplexity = tlist['perplexity']
    if perplexity is None:
      perplexity = get_multiscale_perplexities(2*x_train.shape[0])
      print('Using multiple perplexities: %s' % (','.join(map(str, perplexity))))

    ptSNE = Parametric_tSNE(x_train.shape[1:], num_outputs, perplexity,
                        alpha=alpha_, do_pretrain=do_pretrain, batch_size=batch_size,
                        seed=54321)

    model_path = model_path_template.format(model_tag=tlist['tag'], test_data_tag=test_data_tag)

    if override or not os.path.exists(model_path):
      ptSNE.fit(y_train, epochs=epochs, verbose=1)
      print('{time}: Saving model {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
      ptSNE.save_model(model_path)
    else:
      print('{time}: Loading from {model_path}'.format(time=datetime.datetime.now(), model_path=model_path))
      ptSNE.restore_model(model_path)

    tlist['transformer'] = ptSNE

  pdf_obj = PdfPages(figure_template.format(test_data_tag=test_data_tag))

  for transformer_dict in transformer_list:
    transformer = transformer_dict['transformer']
    tag = transformer_dict['tag']
    label = transformer_dict['label']

    output_res = transformer.transform(train_data)
    test_res = transformer.transform(test_data)

    plt.figure()
    # Create a contour plot of training data
    _plot_kde(output_res, pick_rows, color_palette, 0.5)

    # Scatter plot of test data
    _plot_scatter(test_res, test_pick_rows, color_palette, alpha=0.1, symbol='*')

    leg = plt.legend(bbox_to_anchor=(1.0, 1.0))
    # Set marker to be fully opaque in legend
    for lh in leg.legendHandles:
      lh._legmarker.set_alpha(1.0)

    plt.title('{label:s} Transform with {num_clusters:d} clusters\n{test_data_tag:s} Data'.format(label=label, num_clusters=num_clusters, test_data_tag=test_data_tag.capitalize()))

  if pdf_obj:
    plt.savefig(pdf_obj, format='pdf')

  if pdf_obj:
    pdf_obj.close()
  else:
    plt.show()

main()
