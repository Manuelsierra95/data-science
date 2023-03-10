import imports as im

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
    data_batch = data.map(im.modeling.process_image()).batch(batch_size)

  elif valid_data:
    print('Creating valid data batches...')
    data = im.tf.data.Dataset.from_tensor_slices((im.tf.constant(X), # filepath
                                              im.tf.constant(y))) # labels
    data_batch = data.map(im.modeling.get_image_label()).batch(batch_size)

  else:
    print('Creating training data batches...')
    # Turn filepaths and labels into a Tensor
    data = im.tf.data.Dataset.from_tensor_slices((im.tf.constant(X),
                                              im.tf.constant(y)))
    # Shuffle the pathnames and labels (is faster shuffle labels and then map this images, than shuffle the images)
    data = data.shuffle(buffer_size=len(X))
    data_batch = data.map(im.modeling.get_image_label()).batch(batch_size)
  
  return data_batch

def unbatchify(data):
  '''
  Take a batched dataset of (image, label) Tensor and return 
  separate array of images and labels
  example: val_images, val_labels = unbatchify(val_data)
  '''
  _images = []
  _labels = []
  for image, label in data.unbatch().as_numpy_iterator():
    _images.append(image)
    _labels.append(im.modeling.get_pred_label(label))
  
  return _images, _labels
