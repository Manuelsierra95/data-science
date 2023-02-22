import imports as im

def open_img(image_path):
    img = im.Image.open(image_path)
    print(img.width, img.height)
    im.display(img)

def create_id_labels_path_image(folder_path, labels_id, check_error_len=True):
    files = [folder_path+fname+'.jpg' for fname in labels_id]
    if check_error_len==True:
        if len(im.os.listdir(folder_path)) == len(files):
            print('They have the same len')
        else:
            print('We have a error with the number of files!!')
            print(f'labels items: {len(files)}\nfolder items: {len(im.os.listdir(folder_path))}')
    return files

def labels_to_boolean(label_to_conver, info=True):
    unique_labels = label_to_conver.unique()
    boolean_labels = [label == unique_labels for label in label_to_conver]
    if info==True:
        print(label_to_conver[0])
        print(im.np.where(unique_labels == label_to_conver[0]))
        print(boolean_labels[0].argmax())
        print(boolean_labels[0].astype(int))
    return boolean_labels

