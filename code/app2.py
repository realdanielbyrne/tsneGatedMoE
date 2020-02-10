from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__doc__= """ Example usage of parametric_tSNE.
Generate some simple data in high (14) dimension, train a model,
and run additional generated data through the trained model"""

import sys,os, datetime, shutil, zipfile, glob,math
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Load Keras and PlaidML
import plaidml.keras
from keras.models import Model
from keras.datasets import cifar10
from keras.utils import plot_model
from keras.utils import to_categorical
from keras.layers import Dense, Input, Dropout
from keras.callbacks import Callback
from keras import backend as K
from parametric_tSNE import Parametric_tSNE
from parametric_tSNE.utils import get_multiscale_perplexities


# Globals
plt.style.use('ggplot')
num_classes = 10
cur_path = os.path.realpath(__file__)
_cur_dir = os.path.dirname(cur_path)
_par_dir = os.path.abspath(os.path.join(_cur_dir, os.pardir))
sys.path.append(_cur_dir)
sys.path.append(_par_dir)

num_classes = 10
test_data_tag = 'none'
batch_size = 5000
low_dim = 2
nb_epoch = 20
shuffle_interval = nb_epoch + 1
n_jobs = 4
perplexity = 30.0

color_palette = sns.color_palette("hls", num_classes)
model_path_template = 'ptsne_{model_tag}_{test_data_tag}.h5'
figure_template = 'ptsne_viz_tSNE_{test_data_tag}.pdf'
log_dir = '.\output'

def load_cifar10_data():
  # load the CIFAR10 data
  (x_train, y_train), (x_test, y_test) = cifar10.load_data()
  n, channel, row, col = x_train.shape

  x_train = x_train.reshape(-1, channel * row * col).astype('float32') / 255
  x_test = x_test.reshape(-1, channel * row * col).astype('float32') / 255

  # Convert class vectors to binary class matrices.
  #y_train_cat = to_categorical(y_train, num_classes)
  #y_test_cat = to_categorical(y_test, num_classes)

  return (x_train,x_test),(y_train,y_test)

def main():
    (x_train, y_train), (x_test, y_test) = load_cifar10_data()
    high_dims = x_train.shape[1]
    num_outputs = 2
    perplexity = 30

    ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity)
    ptSNE.fit(x_train)
    output_res = ptSNE.transform(x_train)

main()
