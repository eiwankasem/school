import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plotDecisionBoundary(X1, X2, y, model):
    """
    Plot the decision boundary for a binary classification model along with data points.
    Uses the meshgrid approach from the lecture: classify every point on a dense grid,
    colour the background, then overlay the actual data.

    Parameters:
        X1 (array-like): Feature values for the first feature.
        X2 (array-like): Feature values for the second feature.
        y  (array-like): True binary labels (0 or 1).
        model (object): Trained model with a predict(X) method that returns probabilities.

    Returns:
        None
    """
    X1 = np.array(X1)
    X2 = np.array(X2)
    y  = np.array(y)

    h = 0.02
    x_min, x_max = X1.min() - 0.2, X1.max() + 0.2
    y_min, y_max = X2.min() - 0.2, X2.max() + 0.2

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))

    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = model.predict(grid)
    classes = (probs >= 0.5).reshape(xx.shape)

    cmap_bg   = ListedColormap(['#FFAAAA', '#AAFFAA'])
    cmap_pts  = ListedColormap(['#CC0000', '#006600'])

    plt.pcolormesh(xx, yy, classes, cmap=cmap_bg, shading='auto', alpha=0.6)
    scatter = plt.scatter(X1, X2, c=y, cmap=cmap_pts, marker='.', edgecolors='k',
                          linewidths=0.3, s=20)
    plt.xlabel('Feature 1 (normalized)')
    plt.ylabel('Feature 2 (normalized)')
    handles = [plt.Line2D([0], [0], marker='o', color='w',
                           markerfacecolor=cmap_pts(i), markersize=8, label=lbl)
               for i, lbl in enumerate(['Class 0', 'Class 1'])]
    plt.legend(handles=handles)
