from __future__ import absolute_import, division, print_function, unicode_literals
import sys, os, datetime, shutil, zipfile, glob,math
import numpy as np
import argparse

from parametric_tSNE import Parametric_tSNE
import utils
import argparse

if __name__ == '__main__':
  
  parser = argparse.ArgumentParser(description='Control MLP Classifier')
  parser.add_argument("-c", "--categorical",
                      default=False,
                      help="Convert class vectors to binary class matrices ( One Hot Encoding ).")
  parser.add_argument("-s", "--embedding_type",
                      default='mean',
                      help="embedding_type - sample: Samples a single x_test latent variable for each class\n\
                            mean: Averages all x_test latent variables")

  args = parser.parse_args()

  
  # load data
  (x_train, y_train), (x_test, y_test),num_labels,y_test_cat = utils.load_minst_data(categorical=True)
  input_dim = output_dim = x_train.shape[-1]
  

  high_dims = x_train.shape[1]
  num_outputs = 2
  perplexity = 30
  ptSNE = Parametric_tSNE(high_dims, num_outputs, perplexity)
  ptSNE.fit(x_train)
  output_res = ptSNE.transform(x_train)
