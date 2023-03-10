import imports as im

# Create a function to build a TensorBoard callback
def create_tensorboard_callback(log_path=im.info.LOG_PATH):
  '''
  Creates a log dir
  Edit the info.py file for change the path
  '''
  # Create log dir
  logdir = im.os.path.join(log_path,
                        im.datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
  return im.tf.keras.callbacks.TensorBoard(logdir)

def create_earlystopppig_callback():
    # Create EarlyStopping callback
    earlystopping = im.tf.keras.callbacks.EarlyStopping(monitor='val_accuracy',
                                                 patience=3)
    return earlystopping