import tensorflow as tf
import tensorflow_hub as hub
import info
import callbacks
import numpy as np


def process_image(image_path):
  '''
  Take a image and turn into a Tensor
  '''
  # Read image
  image = tf.io.read_file(image_path)
  # Turn jpg image into Tensor with RGB
  image = tf.image.decode_jpeg(image, channels=3)
  # Conver RGB from 0-255 to 0-1
  image = tf.image.convert_image_dtype(image, tf.float32)
  # Resize the image
  image = tf.image.resize(image, size=[info.IMG_SIZE, info.IMG_SIZE])

  return image

def get_image_label(image_path, label):
  '''
  Return the processed image, with the label associated
  '''
  return process_image(image_path), label

def create_model(input_shape=info.INPUT_SHAPE, output_shape=info.OUTPUT_SHAPE, model_url=info.MODEL_URL):
  '''
  Build a model with from a pre trained model
  Edit info.py file to change parameters
  '''
  print(f'Building model with: {info.model_url}')

  # Setup the layers
  model = tf.keras.Sequential([
    hub.KerasLayer(model_url), # Layer 1 (input layer)
    tf.keras.layers.Dense(units=output_shape,
                          activation='softmax') # Layer 2 (output layer)
  ])
  # Compile the model
  model.compile(
      loss=tf.keras.losses.CategoricalCrossentropy(),
      optimizer=tf.keras.optimizers.Adam(),
      metrics=['accuracy']
  )
  # Build the model
  model.build(input_shape)  # Batch input shape.
  
  return model

def train_model(train_data, val_data, num_epoch=info.NUM_EPOCH):
  '''
  Trains a given model and returns the trained version
  '''
  model = create_model()
  # Create new TensorBoard session everytime we train a model
  tensorboard = callbacks.create_tensorboard_callback()
  earlystopping = callbacks.create_earlystopppig_callback()
  # Fit the model with the callbacks
  model.fit(x=train_data,
            epochs=num_epoch,
            validation_data=val_data,
            validation_freq=1,
            callbacks=[tensorboard, earlystopping])
  
  return model

def check_logs(log_path=info.LOG_PATH):
  # Checking the logs
  log_path = './' + log_path
  %tensorboard --logdir "$log_path"

def get_pred_label(preds_proba):
  '''
  Turns an array of preds proba into a label
  '''
  return info.UNIQUE_LABELS[np.argmax(preds_proba)]

def print_preds_proba(index, preds=None):
  '''
  Prints preds proba of a element in the array of preds
  '''
  if preds == None:
    print('Need to pass the model.predict()')
  else:
    print(preds[index])
    print(f"Max value (proba of pred): {np.max(preds[index])}")
    print(f"Sum: {np.sum(preds[index])}")
    print(f"Pred value index: {np.argmax(preds[index])}")
    print(f"Predicted label: {get_pred_label(preds[index])}")

