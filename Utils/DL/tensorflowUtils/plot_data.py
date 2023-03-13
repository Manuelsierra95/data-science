import imports as im

def open_img(image_path):
    img = im.Image.open(image_path)
    print(img.width, img.height)
    print("shape: ", im.np.array(img).shape)
    im.display(img)

def show_train_images(train_data, unique_labels, figsize=(10, 10), img_stack=None):
    '''
    Show each unique images in train_data
    Also can edit the figsize and the image_stack if the plot show ugly
    '''
    train_images, train_labels = next(train_data.as_numpy_iterator())
    if img_stack == None:
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
    im.plt.title(
        f"pred: {pred_label} {im.np.max(pred_prob)*100:2.0f}% true: {true_label}", color=color)

def plot_labels_proba(preds_proba, labels, n=1):
    '''
    Plot the best 10 preds of a label
    `n` is the index of the preds array
    '''
    pred_prob, true_label = preds_proba[n], labels[n]
    # Find top 10
    top_10_preds_indexes = pred_prob.argsort(
    )[-10:][::-1]  # [::-1] Is for reverse
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
        top_plot[im.np.argmax(top_10_preds_labels ==
                              true_label)].set_color('green')

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

def plot_decision_boundary(model, X, y):
    '''
    Plot the decision boundary created by a model predicting on X
    '''
    # Define the axis boundaries of the plot and create a meshgrid
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
    xx, yy = im.np.meshgrid(im.np.linspace(x_min, x_max, 100),
                            im.np.linspace(y_min, y_max, 100))

    # Create X value (we're going to make predictions on these)
    x_in = im.np.c_[xx.ravel(), yy.ravel()]  # stack 2D arrays together

    # Make predictions
    y_pred = model.predict(x_in)

    # Check for multi-class
    if len(y_pred[0]) > 1:
        print("doing multiclass classification...")
        # We have to reshape our predictions to get them ready for plotting
        y_pred = im.np.argmax(y_pred, axis=1).reshape(xx.shape)
    else:
        print("doing binary classification...")
        y_pred = im.np.round(y_pred).reshape(xx.shape)

    # Plot the decision boundary
    im.plt.contourf(xx, yy, y_pred, cmap=im.plt.cm.RdYlBu, alpha=0.7)
    im.plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=im.plt.cm.RdYlBu)
    im.plt.xlim(xx.min(), xx.max())
    im.plt.ylim(yy.min(), yy.max())

def plot_confusion_matrix(y_true, y_pred, index, columns, color='OrRd', num_fmt='d', figsize=(10,10)):
    '''
    Plot a confusion matrix using Seaborn's heatmap(), form a tensor of predictions
    '''
    # Convert the tensor to a numpy array
    y_pred = [im.np.argmax(y_pred[x], 0) for x in range(len(y_pred))]

    # Create the confusion matrix
    cm = im.confusion_matrix(y_true, y_pred)

    # Create a dataframe from the confusion matrix
    cm_df = im.pd.DataFrame(cm,
                            index=index,
                            columns=columns)
    
    # Plot the confusion matrix
    im.plt.figure(figsize=figsize)
    im.sns.heatmap(cm_df, annot=True, cmap=color, fmt=num_fmt)
    im.plt.title('Confusion Matrix')
    im.plt.ylabel('Actal Values')
    im.plt.xlabel('Predicted Values')
    im.plt.show()
