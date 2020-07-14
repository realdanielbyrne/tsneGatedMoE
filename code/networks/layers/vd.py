from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import backend as K
from tensorflow.python.eager import context

EPSILON = 1e-14
ALPHA = 3.

class VarDropout(Layer):
  def __init__(self,
               num_outputs,
               clip_alpha = 8.,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.clip_alpha = clip_alpha

  def build(self, input_shape):
    kernel_shape = [input_shape[-1], self.num_outputs]

    self.theta = self.add_weight(shape = kernel_shape,
                            initializer=tf.initializers.TruncatedNormal(),
                            regularizer=tf.keras.regularizers.L2(.001),
                            trainable=True,
                            name = 'theta')

    self.log_sigma2 = tf.Variable(tf.zeros_like(self.theta), trainable=False, name='log_sigma2', shape=kernel_shape)
    
    self.step = tf.Variable (0.,
                            name='step', 
                            trainable = False)

  def call(self, inputs, training = None):
    
    if training:
      self.step = self.step.assign_add(.001)
      
      if self.step > 1.:

        log_alpha = self.compute_log_alpha(self.theta, self.log_sigma2)
        self.log_sigma2.assign (self.compute_log_sigma2(self.theta,log_alpha))

        mu = tf.keras.activations.elu(tf.matmul(inputs, self.theta))
        si = tf.sqrt(tf.matmul(tf.square(inputs), tf.exp(log_alpha) * tf.square(self.theta)) + EPSILON)
        x = mu + si * tf.random.normal(tf.shape(si))
        loss = self.negative_dkl(self.theta, self.log_sigma2) / 47000  

      else:
        x = tf.keras.activations.elu(tf.matmul(inputs, self.theta))
        loss = 0.

    else:
      x = self.matmul_eval (inputs, self.theta, self.log_sigma2, self.clip_alpha)
      loss = 0.

    self.add_loss(loss) 

    if not context.executing_eagerly():
      # Set the static shape for the result since it might be lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      x.set_shape(self.compute_output_shape(inputs.shape))

    return x

  def compute_output_shape(self, input_shape):
    return (input_shape[0],self.num_outputs)


  @tf.function
  def matmul_eval(self,x,theta,log_sigma2,threshold):
      # Compute the weight mask by thresholding on
      # the log-space alpha values
      log_alpha = self.compute_log_alpha(theta,log_sigma2)
      weight_mask = tf.cast(tf.less(log_alpha, threshold), tf.float32)
      return tf.matmul(x, theta)

  @tf.function
  def compute_log_sigma2(self,log_alpha, theta):
      return log_alpha + tf.math.log(tf.square(theta) + EPSILON)
  
  @tf.function
  def compute_log_alpha(self,theta,log_sigma2):
    return  tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + EPSILON),-ALPHA,ALPHA)

  @tf.function
  def negative_dkl(self,theta,log_sigma2):
    log_alpha = self.compute_log_alpha(theta,log_sigma2)
    # Constant values for approximating the kl divergence
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -k1
    
    # Compute each term of the KL and combine
    term_1 = k1 * tf.nn.sigmoid(k2 + k3 * log_alpha)
    term_2 = -0.5 * tf.math.log1p(tf.math.exp(tf.math.negative(log_alpha)))
    eltwise_dkl = term_1 + term_2 + c
    return tf.reduce_sum(eltwise_dkl)

  def get_config(self):
      return {
        'num_outputs': self.num_outputs, 
        'clip_alpha': self.clip_alpha,
        'step': self.step
        }