# scratchpad
initial_thetas, initial_log_sigma2s = initial_values
theta_lookup = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(tf.range(num_labels)),
        values=tf.constant(initial_thetas),
    ),
    default_value=tf.constant(0.),
    name="class_weight"
)

log_sigma2_lookup = tf.lookup.StaticHashTable(
    initializer=tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(tf.range(num_labels)),
        values=tf.constant(initial_log_sigma2s),
    ),
    default_value=tf.constant(0.),
    name="class_weight"
)

def compute_log_sigma2(log_alpha, theta, eps=EPSILON):
    return log_alpha + tf.math.log(tf.square(theta) + eps)

def compute_log_alpha(variational_params, eps=EPSILON, value_limit=8.):
  theta, log_sigma2 = variational_params
  log_alpha = log_sigma2 - tf.math.log(tf.square(theta) + eps)

  if value_limit is not None:
    # If a limit is specified, clip the alpha values
    return tf.clip_by_value(log_alpha, -value_limit, value_limit)
  return log_alpha

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, yt))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(BATCH_SIZE)
EPSILON = 1e-8

for epoch in range(EPOCHS):
  print("\nStart of epoch %d" % (epoch,))

  # Iterate over the batches of the dataset.
  for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
    x = x_batch_train
    y = tf.cast(y_batch_train,tf.int32)
    num_outputs = 784

    theta = theta_lookup.lookup(y)
    log_sigma2 = log_sigma2_lookup.lookup(y)
  
    variational_params = (theta, log_sigma2)
    log_alpha = compute_log_alpha(variational_params, eps = 1e-8) 

    clip_alpha = 3
    if clip_alpha is not None:
      #Compute log_sigma2 again so that we can clip on the log alpha magnitudes
      log_sigma2 = compute_log_sigma2(log_alpha, theta, EPSILON)

    kernel_shape = [x.shape[1], num_outputs]

    #if training:
    mu = x * theta
    std = tf.sqrt(tf.square(x)*tf.exp(log_sigma2) + 1e-8)
    noise = tf.random.normal(kernel_shape)
    val = tf.matmul(std, tf.random.normal(kernel_shape))
