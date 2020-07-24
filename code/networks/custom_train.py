# Custom_train
import tensorflow as tf

def vd_loss(model):
  log_alphas = []
  fraction = 0.
  theta_logsigma2 = [layer.variables for layer in model.layers if 'var_dropout' in layer.name]
  for theta, log_sigma2, b, step in theta_logsigma2:
    log_alphas.append(compute_log_alpha(theta, log_sigma2))
  
  fraction = tf.minimum(tf.maximum(fraction,step),1.)
  return log_alphas, fraction

def negative_dkl(log_alpha=None):
  # Constant values for approximating the kl divergence
  k1, k2, k3 = 0.63576, 1.8732, 1.48695
  c = -k1
  # Compute each term of the KL and combine
  term_1 = k1 * tf.nn.sigmoid(k2 + k3*log_alpha)
  term_2 = -0.5 * tf.math.log1p(tf.math.exp(tf.math.negative(log_alpha)))
  eltwise_dkl = term_1 + term_2 + c
  return -tf.reduce_sum(eltwise_dkl)

def custom_train(model, x_train, y_train, optimizer, x_test,y_test, loss_fn):
  # Keep results for plotting
  train_loss_results = []
  train_accuracy_results = []

  # Instantiate an optimizer.
  epoch_loss_avg = tf.keras.metrics.Mean()
  epoch_accuracy = tf.keras.metrics.CategoricalAccuracy()
  val_accuracy = tf.keras.metrics.CategoricalAccuracy()

  # Prepare the validation dataset.
  # Reserve 5,000 samples for validation.
  x_val = x_train[-3240:]
  y_val = y_train[-3240:]
  x_train = x_train[:-3240]
  y_train = y_train[:-3240]
  val_dataset = tf.data.Dataset.from_tensor_slices((x_val, y_val))
  val_dataset = val_dataset.batch(BATCH_SIZE)

  # prepare data  
  train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
  train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)

  #@tf.function
  def train_step(x, y):
    with tf.GradientTape() as tape:
      logits = model(x, training=True)  # Logits for this minibatch

      # Compute the loss value for this minibatch.
      loss_value = tf.nn.softmax_cross_entropy_with_logits(y, logits)

      # Add kld layer losses created during this forward pass:
      log_alphas,dkl_fraction = vd_loss(model)
      dkl_loss = tf.add_n([negative_dkl(log_alpha=a) for a in log_alphas])

      #regularizer intensifies over the course of ramp-up
      tf.summary.scalar('dkl_fraction', dkl_fraction)
      tf.summary.scalar('dkl_loss_gross',dkl_loss )
      dkl_loss = dkl_loss * dkl_fraction
      
      dkl_loss =  sum(model.losses) 
      tf.summary.scalar('dkl_loss_net',dkl_loss)
      #dkl_loss = dkl_loss / float(x_train.shape[0])
      loss_value += dkl_loss
      
      # Retrievethe gradients of the trainable variables with respect to the loss.
      grads = tape.gradient(loss_value, model.trainable_weights)
      # Minimize the loss.
      optimizer.apply_gradients(zip(grads, model.trainable_weights))

      # Update training metrics
      epoch_loss_avg.update_state(loss_value) 
      epoch_accuracy.update_state(y_batch_train,logits)
    return loss_value

  @tf.function
  def test_step(x, y):
      val_logits = model(x, training=False)
      val_accuracy.update_state(y, val_logits)

  for epoch in range(EPOCHS):
    print("\nStart of epoch %d" % (epoch,))
    
    # Iterate over the batches of the dataset.
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):

      # Run the forward pass.
      loss_value = train_step(x_batch_train, y_batch_train)

    # Display metrics at the end of each epoch.
    print("Training acc over epoch: %.4f" % (float(epoch_accuracy.result()),))
    print("Training loss over epoch: %.4f" % (float(epoch_loss_avg.result()),))

    # Reset training metrics at the end of each epoch
    epoch_accuracy.reset_states()      
    epoch_loss_avg.reset_states()

    # Run a validation loop at the end of each epoch.
    for x_batch_val, y_batch_val in val_dataset:
        test_step(x_batch_val, y_batch_val)
    val_acc = val_accuracy.result()
    val_accuracy.reset_states()
    print("Validation acc: %.4f" % (float(val_acc),))

  test_step(x_test, y_test)
  val_acc = val_accuracy.result()
  val_accuracy.reset_states()
  print("Test acc: %.4f" % (float(val_acc),))

