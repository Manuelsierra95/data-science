import imports as im

def open_img(image_path):
    img = im.Image.open(image_path)
    print(img.width, img.height)
    im.display(img)

def show_train_images(train_data, unique_labels, figsize=(10,10), img_stack=None):
  '''
  Show each unique images in train_data
  Also can edit the figsize and the image_stack if the plot show ugly
  '''
  train_images, train_labels = next(train_data.as_numpy_iterator())
  if img_stack==None:
    img_stack = int(len(unique_labels)/5)
  im.plt.figure(figsize=figsize)
  for i in range(img_stack):
    im.plt.subplot(5, 5, i+1)
    im.plt.imshow(train_images[i])
    im.plt.title(unique_labels[train_labels[i].argmax()])
  im.plt.tight_layout(h_pad=0.01)

def plot_preds(preds_proba, labels, images, n=1):
  '''
  View the preds and get the image
  `n` is the index of the preds array
  '''
  pred_prob, true_label, image = preds_proba[n], labels[n], images[n]

  # Get preds
  pred_label = im.modeling.get_pred_label(pred_prob)

  # Change color if preds is rigth or wrong
  color = 'green' if pred_label == true_label else 'red'
  
  # Plot image 
  im.plt.imshow(image)
  im.plt.xticks([])
  im.plt.yticks([])
  im.plt.title(f"pred: {pred_label} {im.np.max(pred_prob)*100:2.0f}% true: {true_label}", color=color)

def plot_labels_proba(preds_proba, labels, n=1):
  '''
  Plot the best 10 preds of a label
  `n` is the index of the preds array
  '''
  pred_prob, true_label = preds_proba[n], labels[n]
  # Find top 10
  top_10_preds_indexes = pred_prob.argsort()[-10:][::-1] # [::-1] Is for reverse
  top_10_preds_values = pred_prob[top_10_preds_indexes]
  top_10_preds_labels = im.info.UNIQUE_LABELS[top_10_preds_indexes]

  # Plot
  top_plot = im.plt.bar(im.np.arange(len(top_10_preds_labels)),
                     top_10_preds_values,
                     color='grey')
  im.plt.xticks(im.np.arange(len(top_10_preds_labels)),
                       labels=top_10_preds_labels,
                       rotation='vertical')
  
  # Change color of true label
  if im.np.isin(true_label, top_10_preds_labels):
    top_plot[im.np.argmax(top_10_preds_labels == true_label)].set_color('green')

def plot_img_graph(preds_proba, pred_label, images, img_stack=0):
  '''
  Plot the images and graphs of preds proba of best 10 unique label
  img_stack get a stack of 6 examples. If img_stack = 0. Examples going 0 to 6
  '''
  print(f'Ploting {img_stack} to {img_stack+6} images of the dataset...')
  num_rows = 3
  num_cols = 2
  num_images = num_rows*num_cols
  im.plt.figure(figsize=(10*num_cols, 5*num_rows))
  for i in range(num_images):
    im.plt.subplot(num_rows, 2*num_cols, 2*i+1)
    plot_preds(preds_proba=preds_proba,
              labels=pred_label,
              images=images,
              n=i+img_stack)
    im.plt.subplot(num_rows, 2*num_cols, 2*i+2)
    plot_labels_proba(preds_proba=preds_proba,
                      labels=pred_label,
                      n=i+img_stack)
  im.plt.tight_layout(h_pad=0.5)
  im.plt.show()