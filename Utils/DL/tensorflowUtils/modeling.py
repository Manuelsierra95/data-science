import imports as im

def process_image(image_path):
  '''
  Take a image and turn into a Tensor
  '''
  # Read image
  image = im.tf.io.read_file(image_path)
  # Turn jpg image into Tensor with RGB
  image = im.tf.image.decode_jpeg(image, channels=3)
  # Conver RGB from 0-255 to 0-1
  image = im.tf.image.convert_image_dtype(image, im.tf.float32)
  # Resize the image
  image = im.tf.image.resize(image, size=[im.info.IMG_SIZE, im.info.IMG_SIZE])

  return image

def get_image_label(image_path, label):
  '''
  Return the processed image, with the label associated
  '''
  return process_image(image_path), label

def create_data_batches(X, y=None, batch_size=im.info.BATCH_SIZE, valid_data=False, test_data=False):
  '''
  Create a batches of data from a image (X) and label (y) pairs
  Shuffles the data if it's training data but dosen't if it's validation data
  Also accepts test data as input (no labels)
  Example:  train_data = create_data_batches(X_train, y_train)
            val_data = create_data_batches(X_val, y_val, valid_data=True)
  '''
  if test_data:
    print('Creating test data batches...')
    data = im.tf.data.Dataset.from_tensor_slices((im.tf.constant(X))) # No labels
    data_batch = data.map(process_image).batch(batch_size)

  elif valid_data:
    print('Creating valid data batches...')
    data = im.tf.data.Dataset.from_tensor_slices((im.tf.constant(X), # filepath
                                              im.tf.constant(y))) # labels
    data_batch = data.map(get_image_label).batch(batch_size)

  else:
    print('Creating training data batches...')
    # Turn filepaths and labels into a Tensor
    data = im.tf.data.Dataset.from_tensor_slices((im.tf.constant(X),
                                              im.tf.constant(y)))
    # Shuffle the pathnames and labels (is faster shuffle labels and then map this images, than shuffle the images)
    data = data.shuffle(buffer_size=len(X))
    data_batch = data.map(get_image_label).batch(batch_size)
  
  return data_batch

def create_model(input_shape=im.info.INPUT_SHAPE, output_shape=im.info.OUTPUT_SHAPE, model_url=im.info.MODEL_URL):
  '''
  Build a model with from a pre trained model
  Edit im.info.py file to change parameters
  '''
  print(f'Building model with: {im.info.model_url}')

  # Setup the layers
  model = im.tf.keras.Sequential([
    im.hub.KerasLayer(model_url), # Layer 1 (input layer)
    im.tf.keras.layers.Dense(units=output_shape,
                          activation='softmax') # Layer 2 (output layer)
  ])
  # Compile the model
  model.compile(
      loss=im.tf.keras.losses.CategoricalCrossentropy(),
      optimizer=im.tf.keras.optimizers.Adam(),
      metrics=['accuracy']
  )
  # Build the model
  model.build(input_shape)  # Batch input shape.
  
  return model

def train_model(train_data, val_data, num_epoch=im.info.NUM_EPOCH):
  '''
  Trains a given model and returns the trained version
  '''
  model = create_model()
  # Create new TensorBoard session everytime we train a model
  tensorboard = im.callbacks.create_tensorboard_callback()
  earlystopping = im.callbacks.create_earlystopppig_callback()
  # Fit the model with the callbacks
  model.fit(x=train_data,
            epochs=num_epoch,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, earlystopping])
  im.ntfy.send_notification(f"Model {model} trained for {num_epoch} epochs")
  
  return model

def check_logs(log_path=im.info.LOG_PATH):
  # Checking the logs
  log_path = './' + log_path
  # %tensorboard --logdir "$log_path"

def get_pred_label(preds_proba):
  '''
  Turns an array of preds proba into a label
  '''
  return im.info.UNIQUE_LABELS[im.np.argmax(preds_proba)]

def print_preds_proba(index, preds):
  '''
  Print preds proba for a given index of a preds array
  '''
  print(preds[index])
  print(f"Max value (proba of pred): {im.np.max(preds[index])}")
  print(f"Sum: {im.np.sum(preds[index])}")
  print(f"Pred value index: {im.np.argmax(preds[index])}")
  print(f"Predicted label: {get_pred_label(preds[index])}")