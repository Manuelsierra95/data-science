from PIL import Image
import matplotlib.pyplot as plt
import modeling
import numpy as np
import info

def open_img(image_path):
    img = Image.open(image_path)
    print(img.width, img.height)
    display(img)

def show_train_images(train_data, unique_labels, figsize=(10,10), img_stack=None):
  '''
  Show each unique images in train_data
  Also can edit the figsize and the image_stack if the plot show ugly
  '''
  train_images, train_labels = next(train_data.as_numpy_iterator())
  if img_stack==None:
    img_stack = int(len(unique_labels)/5)
  plt.figure(figsize=figsize)
  for i in range(img_stack):
    plt.subplot(5, 5, i+1)
    plt.imshow(train_images[i])
    plt.title(unique_labels[train_labels[i].argmax()])
  plt.tight_layout(h_pad=0.01)

def plot_preds(preds_proba, labels, images, n=1):
  '''
  View the preds and get the image
  `n` is the index of the preds array
  '''
  pred_prob, true_label, image = preds_proba[n], labels[n], images[n]

  # Get preds
  pred_label = modeling.get_pred_label(pred_prob)

  # Change color if preds is rigth or wrong
  color = 'green' if pred_label == true_label else 'red'
  
  # Plot image 
  plt.imshow(image)
  plt.xticks([])
  plt.yticks([])
  plt.title(f"pred: {pred_label} {np.max(pred_prob)*100:2.0f}% true: {true_label}", color=color)


def plot_labels_proba(preds_proba, labels, n=1):
  '''
  Plot the best 10 preds of a label
  `n` is the index of the preds array
  '''
  pred_prob, true_label = preds_proba[n], labels[n]
  # Find top 10
  top_10_preds_indexes = pred_prob.argsort()[-10:][::-1] # [::-1] Is for reverse
  top_10_preds_values = pred_prob[top_10_preds_indexes]
  top_10_preds_labels = info.UNIQUE_LABELS[top_10_preds_indexes]

  # Plot
  top_plot = plt.bar(np.arange(len(top_10_preds_labels)),
                     top_10_preds_values,
                     color='grey')
  plt.xticks(np.arange(len(top_10_preds_labels)),
                       labels=top_10_preds_labels,
                       rotation='vertical')
  
  # Change color of true label
  if np.isin(true_label, top_10_preds_labels):
    top_plot[np.argmax(top_10_preds_labels == true_label)].set_color('green')