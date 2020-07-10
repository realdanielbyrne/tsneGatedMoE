import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import layers

class  ProbabilityDropout(layers.Layer):
  def __init__( self,
                **kwargs):
    super(ProbabilityDropout, self).__init__(**kwargs)

  def call(self, inputs, training = None):
    z, x = inputs
    
    if training :
      mu,var = tf.nn.moments(z,axes=0,keepdims=True) 

      epsilon = tf.random.normal(shape=tf.shape(x))
      z = mu + tf.exp(0.5 * var) * epsilon

      # computes drop probability
      def dropprob(p):
        p_range = [K.min(p,axis = -1),K.max(p, axis = -1)]
        p = tf.nn.softmax(tf.cast(tf.histogram_fixed_width(p, p_range, nbins = self.num_outputs),dtype=tf.float32))
        return p

      probs = tf.map_fn(dropprob, z)

      # push values that are close to zero, too zero, promotes sparse models which are more efficient
      condition = tf.less(probs,self.zero_point)
      probs = tf.where(condition,tf.zeros_like(probs),probs)
      
      # scales output after zeros to encourage sum to be similar to sum before zeroing out connections
      return x * probs
    return x
      
  def compute_output_shape(self, input_shape):
    _, _, in_shape = input_shape
    return in_shape[-1]
