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
               kernel_initializer = tf.keras.initializers.RandomNormal,
               bias_initializer = tf.keras.initializers.RandomNormal,
               kernel_regularizer = tf.keras.regularizers.l1,
               bias_regularizer =  tf.keras.regularizers.l1,
               log_sigma2_initializer = None,
               activity_regularizer = None,
               use_bias=True,
               trainable = True,
               eps=EPSILON,
               threshold=3.,
               clip_alpha=8.,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.kernel_initializer = kernel_initializer
    self.bias_initializer = bias_initializer
    self.kernel_regularizer = kernel_regularizer,
    self.bias_regularizer = bias_regularizer,
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
                            trainable=True)

    if self.log_sigma2_initializer is None:
      #self.log_sigma2_initializer = tf.constant_initializer(value=-10, dtype=tf.float32)
      self.log_sigma2_initializer = tf.random_uniform_initializer()

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            trainable=True)

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = self.bias_initializer,
                            trainable = True)
    else:
      self.b = None
    self.built = True

  def call(self,inputs, training = None):
    if training:
      x = matmul_train(
          inputs, (self.w, self.log_sigma2), clip_alpha=self.clip_alpha)
    else:
      x = matmul_eval(
          inputs, (self.w, self.log_sigma2), threshold=self.threshold)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.b)
    if self.activation is not None:
      return self.activation(x)

    self.add_loss(variational_dropout_dkl_loss([self.w, self.log_sigma2]))
    return x
'''
class ConvVarDropout(layers.Layer):
  def __init__(self,
               num_outputs,
               kernel_size,
               strides,
               padding,
               activation = tf.keras.activations.relu,
               kernel_initializer = tf.keras.initializers.RandomNormal,
               bias_initializer = tf.keras.initializers.RandomNormal,
               kernel_regularizer = tf.keras.regularizers.l1,
               bias_regularizer =  tf.keras.regularizers.l1,
               is_training=True,
               trainable=True,
               use_bias=False,
               eps=common.EPSILON,
               threshold=3.,
               clip_alpha=8.,
               **kwargs):
    super(ConvVarDropout, self).__init__(
        trainable=trainable,
        **kwargs)
    self.num_outputs = num_outputs
    self.kernel_size = kernel_size
    self.strides = [1, strides[0], strides[1], 1]
    self.padding = padding.upper()
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
    dims = input_shape[3]
    kernel_shape = [
        self.kernel_size[0], self.kernel_size[1], dims, self.num_outputs
    ]

    self.w = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            trainable=True)

    if self.log_sigma2_initializer is None:
      #self.log_sigma2_initializer = tf.constant_initializer(value=-10, dtype=tf.float32)
      self.log_sigma2_initializer = tf.random_uniform_initializer()

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            trainable=True)

    if self.use_bias:
      self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer = self.bias_initializer,
                            trainable = True)
    else:
      self.b = None
    self.built = True


  def call(self, inputs):

    if self.is_training:
      output = conv2d_train(
          x=inputs,
          variational_params=(self.kernel, self.log_sigma2),
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          clip_alpha=self.clip_alpha,
          eps=self.eps)
    else:
      output = conv2d_eval(
          x=inputs,
          variational_params=(self.kernel, self.log_sigma2),
          strides=self.strides,
          padding=self.padding,
          data_format=self.data_format,
          threshold=self.threshold,
          eps=self.eps)

    if self.use_bias:
      output = tf.nn.bias_add(output, self.bias)
    if self.activation is not None:
      return self.activation(output)
    else:
      return output
'''
def matmul_eval(
      x,
      variational_params,
      threshold=3.0,
      eps=EPSILON):
    # We expect a 2D input tensor, as is standard in fully-connected layers
    x.get_shape().assert_has_rank(2)
    assert(len(variational_params) == 2)
    assert(variational_params[0].shape == variational_params[1].shape)

    w, log_sigma2 = variational_params

    # Compute the weight mask by thresholding on
    # the log-space alpha values
    log_alpha = compute_log_alpha(log_sigma2, w, eps, value_limit=None)
    weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

    return tf.matmul(
        x,
        w * weight_mask)

def matmul_train(
      x,
      variational_params,
      clip_alpha=3.,
      eps=EPSILON):

    # We expect a 2D input tensor, as in standard in fully-connected layers
    x.get_shape().assert_has_rank(2)
    assert(len(variational_params) == 2)
    assert(variational_params[0].shape == variational_params[1].shape)
    w, log_sigma2 = variational_params

    if clip_alpha is not None:
      # Compute the log_alphas and then compute the
      # log_sigma2 again so that we can clip on the
      # log alpha magnitudes
      log_alpha = compute_log_alpha(log_sigma2, w, eps, clip_alpha)
      log_sigma2 = compute_log_sigma2(log_alpha, w, eps)

    # Compute the mean and standard deviation of the distributions over the
    # activations
    mu = tf.matmul(x, w)
    std_activation = tf.sqrt(tf.matmul(tf.square(x),tf.exp(log_sigma2)) + eps)

    output_shape = tf.shape(std_activation)
    return mu + std_activation * tf.random.normal(output_shape)

def compute_log_sigma2(log_alpha, w, eps=EPSILON):
    return log_alpha + tf.math.log(tf.square(w) + eps)

def compute_log_alpha(log_sigma2, w, eps=EPSILON, value_limit=8.):
    log_alpha = log_sigma2 - tf.math.log(tf.square(w) + eps)

    if value_limit is not None:
      # If a limit is specified, clip the alpha values
      return tf.clip_by_value(log_alpha, -value_limit, value_limit)
    return log_alpha

def negative_dkl(variational_params,
                 clip_alpha=3.,
                 eps=EPSILON,
                 log_alpha=None):

  w, log_sigma2 = variational_params
  log_alpha = compute_log_alpha(log_sigma2, w, eps, clip_alpha)

  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1

  # Compute each term of the KL and combine
  term_1 = k1 * tf.nn.sigmoid(k2 + k3*log_alpha)
  term_2 = -0.5 * tf.log1p(tf.exp(tf.negative(log_alpha)))
  eltwise_dkl = term_1 + term_2 + c
  return -tf.reduce_sum(eltwise_dkl)

def variational_dropout_dkl_loss(variational_params,
                                 start_reg_ramp_up=0.,
                                 end_reg_ramp_up=1000.,
                                 warm_up=True):

  # Calculate the kl-divergence weight for this iteration
  step = tf.train.get_or_create_global_step()
  current_step_reg = tf.maximum(0.0,tf.cast(step - start_reg_ramp_up, tf.float32))
  fraction = tf.minimum(current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)

  dkl_loss = tf.add_n([negative_dkl(variational_params) for a in variational_params])

  if warm_up:
    reg_scalar = fraction * 1

  tf.summary.scalar('reg_scalar', reg_scalar)
  dkl_loss = reg_scalar * dkl_loss

  return dkl_loss
