from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import backend as K
from tensorflow.python.eager import context

EPSILON = 1e-8

class ConstantGausianDropout(Layer):
  def __init__(self,
               num_outputs,
               initial_values,
               activation = tf.keras.activations.relu,
               use_bias = True,
               threshold=3.,
               clip_alpha=None,
               **kwargs):

    super(ConstantGausianDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.use_bias = use_bias
    self.threshold = threshold
    self.clip_alpha = clip_alpha

    # unpack variational parameters, and extrapolate the number of classes for the lookup  
    initial_thetas, initial_log_sigma2s = initial_values
    num_labels = len(initial_thetas)

    # define static lookups for pre-calculated datasets
    theta_lookup = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tf.range(num_labels)),
            values=tf.constant(initial_thetas),
        ),
        default_value=tf.constant(0.),
        name="class_weight"
    )
    self.theta_lookup = theta_lookup

    log_sigma2_lookup = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(tf.range(num_labels)),
            values=tf.constant(initial_log_sigma2s),
        ),
        default_value=tf.constant(0.),
        name="class_weight"
    )
    self.log_sigma2_lookup = log_sigma2_lookup

  def call(self, inputs, training = None):
    x,y = inputs
    y = tf.cast(y,tf.int32)
    num_outputs = self.num_outputs

    theta = self.theta_lookup.lookup(y)
    log_sigma2 = self.log_sigma2_lookup.lookup(y)
    
    # repack parameters for convience 
    variational_params = (theta, log_sigma2)

    # compute dropout rate
    log_alpha = compute_log_alpha(variational_params, eps = EPSILON) 

    # Compute log_sigma2 again so that we can clip on the log alpha magnitudes
    if self.clip_alpha is not None:
      log_sigma2 = compute_log_sigma2(log_alpha, theta, EPSILON)

    if training:
      mu = x * theta
      std = tf.sqrt(tf.square(x) * tf.exp(log_sigma2) + EPSILON)

      kernel_shape = [x.shape[1], num_outputs]
      val = mu + tf.matmul(std, tf.random.normal(kernel_shape))        

    else:
      log_alpha = compute_log_alpha(variational_params, eps = EPSILON) 
      weight_mask = tf.cast(tf.less(log_alpha, self.threshold), tf.float32)
      val = tf.matmul(x,theta * weight_mask)  

    # Apply an activation function to the output
    if self.activation is not None:
      val = self.activation(val)
    
    # Hack to fix tensorflow 2.0 bug that does not call self.compute_output_shape on its own
    return tf.reshape(val, self.compute_output_shape(tf.shape(x)))

  def get_config(self):
    return {
      "num_outputs" : self.num_outputs,
      "clip_alpha" : self.clip_alpha,
      "threshold" : self.threshold,
      "theta_lookup" : self.theta_lookup,
      "log_sigma2_lookup" : self.log_sigma2_lookup,
    }
      
  def compute_output_shape(self, input_shape):
    return  [input_shape.shape[0], self.num_outputs]
    
class VarDropout(Layer):
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

    self.theta = self.add_weight(shape = kernel_shape,
                            initializer=self.kernel_initializer,
                            trainable=True)

    if self.log_sigma2_initializer is None:
      #self.log_sigma2_initializer = tf.constant_initializer(value=-10)
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

  def call(self, inputs, training = None):
    
    if training:
      x = matmul_train (
        inputs, (self.theta, self.log_sigma2), clip_alpha = self.clip_alpha)
    else:
      x = matmul_eval (
        inputs, (self.theta, self.log_sigma2), threshold = self.threshold)

    if self.use_bias:
      x = tf.nn.bias_add(x, self.b)
    
    if self.activation is not None:
      x = self.activation(x)
    
    loss = variational_dropout_dkl_loss([ self.theta, self.log_sigma2 ])
    self.add_loss(loss)

    is_executing_eagerly = context.executing_eagerly()
    if not is_executing_eagerly:
      x.set_shape(x.get_shape())

    return x
    

  def compute_output_shape(self, input_shape):
    return  [input_shape.shape[0], self.num_outputs]

def matmul_eval(
      x,
      variational_params,
      threshold=3.0,
      eps=EPSILON):
    # We expect a 2D input tensor, as is standard in fully-connected layers
    x.get_shape().assert_has_rank(2)
    assert(len(variational_params) == 2)
    assert(variational_params[0].shape == variational_params[1].shape)

    theta, log_sigma2 = variational_params

    # Compute the weight mask by thresholding on
    # the log-space alpha values
    log_alpha = compute_log_alpha(variational_params, eps=eps, value_limit=None)
    weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)

    return tf.matmul(
        x,
        theta * weight_mask)

def matmul_train(
      x,
      variational_params,
      clip_alpha=None,
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
      log_alpha = compute_log_alpha(variational_params, eps, clip_alpha)
      log_sigma2 = compute_log_sigma2(variational_params, eps)

    # Compute the mean and standard deviation of the distributions over the
    # activations
    mu = tf.matmul(x, w)
    std = tf.sqrt(tf.matmul(tf.square(x),tf.exp(log_sigma2)) + eps)

    output_shape = tf.shape(std)
    return mu + std * tf.random.normal(output_shape)

def compute_log_sigma2(log_alpha, theta, eps=EPSILON):
    return log_alpha + tf.math.log(tf.square(theta) + eps)

def compute_log_alpha(variational_params, eps=EPSILON, value_limit=8.):
  theta, log_sigma2 = variational_params
  log_alpha = log_sigma2 - tf.math.log(tf.square(theta) + eps)

  if value_limit is not None:
    # If a limit is specified, clip the alpha values
    return tf.clip_by_value(log_alpha, -value_limit, value_limit)
  return log_alpha

def negative_dkl(variational_params,
                 clip_alpha=3.,
                 eps=EPSILON):
  log_alpha = compute_log_alpha(variational_params, eps, clip_alpha)

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
                                 end_reg_ramp_up=10000.,
                                 warm_up=True):

  # Calculate the kl-divergence weight for this iteration
  step = tf.cast(tf.compat.v1.train.get_or_create_global_step(),tf.float32)
  current_step_reg = step - tf.cast(start_reg_ramp_up,tf.float32)
  current_step_reg = tf.maximum(0.,current_step_reg)
  fraction = tf.minimum(current_step_reg / (end_reg_ramp_up - start_reg_ramp_up), 1.0)

  dkl_loss = tf.add_n([negative_dkl(a) for a in variational_params])

  if warm_up:
    reg_scalar = fraction * 1

  tf.summary.scalar('reg_scalar', reg_scalar)
  dkl_loss = reg_scalar * dkl_loss

  return dkl_loss
