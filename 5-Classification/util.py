import matplotlib.pyplot as plt
import numpy as np
import itertools


def generate_chart():
    X_min = np.min(X[y!=2][:, 0])
    X_max = np.max(X[y!=2][:, 0])
    x = np.linspace(X_min, X_max) # primera dimension del vector

    plt.figure(figsize=(5,5), dpi=150)
    plt.scatter(X[y==0][:10, 0], X[y==0][:10, 1], s=20, label='clase 1', color='darkblue')
    plt.scatter(X[y==1][:10, 0], X[y==1][:10, 1], s=20, label='clase 2', color='darkgreen')

    recta = x*0.5 + 1.75
    plt.plot(x, recta, 'k', label=r'H')
    plt.fill_between(x, y1=yc+0.6, y2=yc-0.6, color='k', alpha=0.2, label='Ancho')

    plt.legend(loc='lower right')
    plt.savefig('./img/svm_0.png', format='png')
    plt.show()
    
    
def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    
    # Plot the training points
    ax.scatter(X[:, 0], X[:, 1], c=y, s=30, cmap=cmap,
               clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),
                         np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,
                           levels=np.arange(n_classes + 1) - 0.5,
                           cmap=cmap,
                           zorder=1)

    ax.set(xlim=xlim, ylim=ylim)
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()