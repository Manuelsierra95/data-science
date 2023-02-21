import datetime
import os
import info
import tensorflow as tf

# TensorBoard notebook extension
%load_ext tensorboard



# Create a function to build a TensorBoard callback
def create_tensorboard_callback(log_path=info.LOG_PATH):
  '''
  Creates a log dir
  Edit the info.py file for change the path
  '''
  # Create log dir
  logdir = os.path.join(log_path,
                        datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  return tf.keras.callbacks.TensorBoard(logdir)

def create_earlystopppig_callback():
    # Create EarlyStopping callback
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=3)
    return earlystopping