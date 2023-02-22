import imports as im

def save_model(model, path_dir=im.os.getcwd(), suffix=None):
  '''
  Save the model in a dir
  '''
  # Create folder
  path_dir+='/models'
  if im.os.path.exists(path_dir) == False:
    im.os.mkdir(path_dir)
  
  # Create model dir pathname with current time
  modeldir = im.os.path.join(path_dir,
                          im.datetime.datetime.now().strftime("%Y%m%d-%H%M%s"))
  model_path = modeldir + '-' + suffix + '.h5' # the format model
  print(f'Saving model to {model_path}...')
  model.save(model_path)
  return model_path

def load_model(model_path):
  '''
  Load a model from a specified path
  '''
  print(f'Loading model from {model_path}...')
  model = im.tf.keras.models.load_model(model_path,
                                     custom_objects={'KerasLayer':im.hub.KerasLayer})
  return model