from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Dropout, Activation, Layer, Add, Concatenate
from tensorflow.keras import backend as K
from tensorflow.python.eager import context

EPSILON = 1e-8
ALPHA = 8.

class VarDropout(Layer):
  def __init__(self,
               num_outputs,
               thresh = 3.,
               activation = tf.keras.activations.relu,
               train_clip = True,
               **kwargs):
    super(VarDropout, self).__init__(**kwargs)
    self.num_outputs = num_outputs
    self.thresh = thresh
    self.activation = activation
    self.train_clip = train_clip

  def build(self, input_shape):
    kernel_shape = [input_shape[-1], self.num_outputs]

    self.theta = self.add_weight(shape = kernel_shape,
                            initializer=tf.initializers.TruncatedNormal(),
                            #regularizer=tf.keras.regularizers.L2(.001),
                            trainable=True,
                            name = 'theta')

    self.b = self.add_weight(shape = (self.num_outputs,),
                            initializer=tf.initializers.TruncatedNormal(),
                            #regularizer=tf.keras.regularizers.L2(.001),
                            trainable=True,
                            name = 'theta')                            

    self.log_sigma2 = self.add_weight(shape = kernel_shape,
                            initializer=tf.initializers.Constant(-10.),
                            #regularizer=tf.keras.regularizers.L2(.001),
                            trainable=True,
                            name = 'log_sigma2')
    
    self.step = tf.Variable (0.,
                            name='step', 
                            trainable = False)

  def call(self, inputs, training = None):
    
    log_alpha = tf.clip_by_value(self.log_sigma2 - tf.math.log(tf.square(self.theta) + EPSILON),-ALPHA,ALPHA)
    clip_mask = tf.less(log_alpha, self.thresh)
    self.step = self.step.assign_add(.001)


    if not training:
      clip_mask = tf.cast(clip_mask,tf.float32)
      x =  tf.matmul(inputs, self.theta * clip_mask)
      loss = 0.

    else:
      if self.step > 1.:
        
        theta = tf.identity(self.theta)
        if self.train_clip:
          theta = tf.where(clip_mask,theta, 0.)

        mu = tf.matmul(inputs, theta)
        si = tf.sqrt(tf.matmul(tf.square(inputs), tf.exp(log_alpha) * tf.square(theta)) + EPSILON)
        int_step = tf.cast(self.step,tf.int32)

        x = mu + si * tf.random.normal(tf.shape(mu))

        # add loss      
        loss = self.negative_dkl(theta, self.log_sigma2) / 49500  

      else:
        x = tf.matmul(inputs, self.theta)
        loss = 0.

    self.add_loss(loss) 

    if not context.executing_eagerly():
      # Set the static shape for the result since it might be lost during array_ops
      # reshape, eg, some `None` dim in the result could be inferred.
      x.set_shape(self.compute_output_shape(inputs.shape))

    return self.activation(x + self.b)

  def compute_output_shape(self, input_shape):
    return (input_shape[0],self.num_outputs)


  @tf.function
  def negative_dkl(self,theta,log_sigma2):
    # Constant values for approximating the kl divergence
    k1, k2, k3 = 0.63576, 1.8732, 1.48695
    c = -k1
    
    log_alpha = tf.clip_by_value(log_sigma2 - tf.math.log(tf.square(theta) + EPSILON),-ALPHA,ALPHA)
    dkl = k1*tf.nn.sigmoid(k2 + k3 * log_alpha) - 0.5*tf.math.log1p(tf.exp(tf.math.negative(log_alpha))) + c
    return -tf.reduce_sum(dkl)

  def get_config(self):
      return {
        'num_outputs': self.num_outputs, 
        'thresh': self.thresh,
        #'step': self.step,
        'activation':self.activation,
        'train_clip':self.train_clip
        }

