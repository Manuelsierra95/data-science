import os
import numpy as np


def create_id_labels_path_image(folder_path, labels_id, check_error_len=True):
    '''
    Join the id column with the folder path of the images.
    Also if `check_error_len == True` print folder len and files len
    '''
    files = [folder_path+fname+'.jpg' for fname in labels_id]
    if check_error_len==True:
        if len(os.listdir(folder_path)) == len(files):
            print('They have the same len')
        else:
            print('We have a error with the number of files!!')
            print(f'labels items: {len(files)}\nfolder items: {len(os.listdir(folder_path))}')
    return files

def labels_to_boolean(label_to_conver, info=True):
    '''
    Convert the label to a boolean matrix
    Also if `info == True` print some info
    '''
    unique_labels = label_to_conver.unique()
    boolean_labels = [label == unique_labels for label in label_to_conver]
    if info==True:
        print(label_to_conver[0])
        print(np.where(unique_labels == label_to_conver[0]))
        print(boolean_labels[0].argmax())
        print(boolean_labels[0].astype(int))
    return boolean_labels



