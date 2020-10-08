import numpy as np
import matplotlib.pyplot as plt

def plot_feature_importances(feature_importances,title,feature_names):
    feature_importances = 100.0*(feature_importances/max(feature_importances))
    index_sorted = np.flipud(np.argsort(feature_importances))
    pos = np.arange(index_sorted.shape[0])+0.5
    plt.figure(figsize=(16,4))
    plt.bar(pos,feature_importances[index_sorted],align='center')
    #plt.xticks(pos,feature_names[index_sorted])
    plt.ylabel('Relative Importance')
    plt.title(title)
    plt.show()