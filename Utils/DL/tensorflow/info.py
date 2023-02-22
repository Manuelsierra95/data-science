import imports as im

'''
    This is a file for put and extract the info of the GLOBAL_VARS
'''
BATCH_SIZE = 32

IMG_SIZE = 224

UNIQUE_LABELS = labels['breed'].unique() # In this case the unique tipe of dog breed

# Setup input shape to the model
INPUT_SHAPE = [None, IMG_SIZE, IMG_SIZE, 3] # None is batch, img_height, img_width, and 3 for RGB

# Setup output shape of our model (get -> len(label.unique()))
OUTPUT_SHAPE = len(UNIQUE_LABELS)

# Setup model URL from TensorFlow Hub
MODEL_URL = 'https://tfhub.dev/google/imagenet/mobilenet_v3_large_075_224/classification/5'

LOG_PATH = '/content/drive/MyDrive/Dog-Vision' + '/log'

NUM_EPOCH = 100


def get_version_info():
    print(im.tf.__version__)
    print(im.hub.__version__)
    print('GPU', 'available :)' if im.tf.config.list_physical_devices("GPU") else 'Not available :(')
