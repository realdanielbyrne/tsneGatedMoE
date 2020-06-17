from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K
EPSILON = 1e-8

class VarDropout(layers.Layer):
  def __init__(self,
               num_outputs,
               activation = tf.keras.activations.relu,
               kernel_initializer = tf.keras.initializers.random_normal,
               bias_initializer = tf.keras.initializers.random_normal,
               kernel_regularizer = tf.keras.regularizers.L1,
               bias_regularizer =  tf.keras.regularizers.L1,
               log_sigma2_initializer = None,
               activity_regularizer = None,
               trainable=True,
               use_bias=True,
               eps=EPSILON,
               threshold=3.,
               clip_alpha=8.,
               name="VarDropout",
               **kwargs):
    super(VarDropout, self).__init__(
        trainable=trainable,
        name=name,
        activity_regularizer=activity_regularizer,
        **kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer
    self.bias_regularizer = bias_regularizer
    self.log_sigma2_initializer = log_sigma2_initializer
    self.use_bias = use_bias
    self.eps = eps
    self.threshold = threshold
    self.clip_alpha = clip_alpha

  def build(self, input_shape):
    input_shape = input_shape.as_list()
    input_hidden_size = input_shape[1]
    kernel_shape = [input_hidden_size, self.num_outputs]

    self.w = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            trainable=True)

    if self.log_sigma2_initializer is None:
      self.log_sigma2_initializer = tf.constant_initializer(value=-10)

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            regularizer=self.kernel_regularizer,
                            trainable=True)

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = self.bias_initializer,
                            regularizer = self.bias_regularizer,
                            trainable = True)
    else:
      self.b = None
    self.built = True

  def call(self,inputs, training = None):
    if training:
          x = VarDropout.matmul_train(
          inputs, (self.w, self.log_sigma2), clip_alpha=self.clip_alpha)
    else:
      x = VarDropout.matmul_eval(
          inputs, (self.w, self.log_sigma2), threshold=self.threshold)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.b)
    if self.activation is not None:
      return self.activation(x)
    return x

  @staticmethod
  def matmul_eval(
      x,
      variational_params,
      transpose_a=False,
      transpose_b=False,
      threshold=3.0,
      eps=EPSILON):
    # We expect a 2D input tensor, as is standard in fully-connected layers
    x.get_shape().assert_has_rank(2)

    theta, log_sigma2 = VarDropout.verify_variational_params(
        variational_params)

    # Compute the weight mask by thresholding on
    # the log-space alpha values
    log_alpha = VarDropout.compute_log_alpha(log_sigma2, theta, eps, value_limit=None)
    weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

    return tf.matmul(
        x,
        theta * weight_mask,
        transpose_a=transpose_a,
        transpose_b=transpose_b)

  @staticmethod
  def matmul_train(
      x,
      variational_params,
      transpose_a=False,
      transpose_b=False,
      clip_alpha=None,
      eps=EPSILON):

    # We expect a 2D input tensor, as in standard in fully-connected layers
    x.get_shape().assert_has_rank(2)

    theta, log_sigma2 = VarDropout.verify_variational_params(
        variational_params)

    if clip_alpha is not None:
      # Compute the log_alphas and then compute the
      # log_sigma2 again so that we can clip on the
      # log alpha magnitudes
      log_alpha = VarDropout.compute_log_alpha(log_sigma2, theta, eps, clip_alpha)
      log_sigma2 = VarDropout.compute_log_sigma2(log_alpha, theta, eps)

    # Compute the mean and standard deviation of the distributions over the
    # activations
    mu_activation = tf.matmul(
        x,
        theta,
        transpose_a=transpose_a,
        transpose_b=transpose_b)
    std_activation = tf.sqrt(tf.matmul(
        tf.square(x),
        tf.exp(log_sigma2),
        transpose_a=transpose_a,
        transpose_b=transpose_b) + eps)

    output_shape = tf.shape(std_activation)
    return mu_activation + std_activation * K.random_normal(output_shape)

  @staticmethod
  def verify_variational_params(variational_params):
    if len(variational_params) != 2:
      raise RuntimeError("Incorrect number of variational parameters.")
    if variational_params[0].shape != variational_params[1].shape:
      raise RuntimeError("Variational parameters must be the same shape.")
    return variational_params

  @staticmethod
  def compute_log_alpha(log_sigma2, theta, eps=EPSILON, value_limit=8.):
    log_alpha = log_sigma2 - tf.math.log(tf.square(theta) + eps)

    if value_limit is not None:
      # If a limit is specified, clip the alpha values
      return tf.clip_by_value(log_alpha, -value_limit, value_limit)
    return log_alpha

  @staticmethod
  def compute_log_sigma2(log_alpha, theta, eps=EPSILON):
    return log_alpha + tf.math.log(tf.square(theta) + eps)
