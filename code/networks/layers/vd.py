from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import backend as K
from tensorflow.python.eager import context

EPSILON = 1e-10
class VarDropout(Layer):
  def __init__(self,
               num_outputs,
               activation = tf.keras.activations.relu,
               clip_alpha = 8.,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.activation = activation
    self.clip_alpha = tf.constant(clip_alpha)
    self.step = tf.Variable(0.,name='step')

  def build(self, input_shape):
    kernel_shape = [input_shape[-1], self.num_outputs]

    self.theta = self.add_weight(shape = kernel_shape,
                            initializer='random_normal',
                            trainable=True,
                            name = 'theta')

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=tf.initializers.GlorotUniform(),
                            trainable=True,
                            name = 'log_sigma2')
    self.built = True

  def call(self, inputs, training = None):
    
    if training:
      x = self.matmul_train (
        inputs, self.theta, self.log_sigma2, self.clip_alpha)
    else:
      x = self.matmul_eval (
        inputs, self.theta, self.log_sigma2, self.clip_alpha)
    
    if self.activation:
      x = self.activation(x)
    
    loss = negative_dkl(self.theta,self.log_sigma2)
    self.add_loss(loss)

    if not context.executing_eagerly():
      # Set the static shape for the result since it might be lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      x.set_shape(self.compute_output_shape(inputs.shape))

    return x

  def compute_output_shape(self, input_shape):
    return (input_shape[0],self.num_outputs)


  def matmul_train(self,x,theta,log_sigma2,clip_alpha):

      if clip_alpha:
        # Compute the log_alphas and then compute the
        # log_sigma2 again so that we can clip on the
        # log alpha magnitudes
        log_alpha = self.compute_log_alpha(theta,log_sigma2)

        #log_alpha = tf.clip_by_value(log_alpha, -value_limit, clip_alpha)
        log_alpha = tf.cast(tf.less(log_alpha, clip_alpha), tf.float32)
        log_sigma2 = self.compute_log_sigma2(theta,log_alpha)

      mu = tf.matmul(x, theta)
      std = tf.sqrt(tf.matmul(tf.square(x),tf.exp(log_sigma2)) + EPSILON)

      output_shape = tf.shape(std)
      return mu + std * tf.random.normal(output_shape)

  def matmul_eval(self,x,theta,log_sigma2,threshold):
      # Compute the weight mask by thresholding on
      # the log-space alpha values
      log_alpha = self.compute_log_alpha(theta,log_sigma2)
      weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)
      return tf.matmul(x, theta * weight_mask)

  @tf.function
  def compute_log_sigma2(self,log_alpha, theta):
      return log_alpha + tf.math.log(tf.square(theta) + EPSILON)
  
  @tf.function
  def compute_log_alpha(self,theta,log_sigma2):
    return  log_sigma2 - tf.math.log(tf.square(theta) + EPSILON)

def compute_log_alpha(theta,log_sigma2):
  return  log_sigma2 - tf.math.log(tf.square(theta) + EPSILON)

def negative_dkl(theta,log_sigma2):
  log_alpha = compute_log_alpha(theta,log_sigma2)
  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1
  
  # Compute each term of the KL and combine
  term_1 = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha)
  term_2 = -0.5 * tf.math.log1p(tf.math.exp(tf.math.negative(log_alpha)))
  eltwise_dkl = term_1 + term_2 + c
  return tf.reduce_sum(eltwise_dkl)
