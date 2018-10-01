import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import itertools

class Plot(object):

    @staticmethod
    def plot_confusion_matrix(cm, classes,
                              normalize=True,
                              title='Matriz de confusão',
                              cmap=plt.cm.Reds):

        plt.figure()
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        plt.imshow(cm, interpolation='nearest', cmap=cmap, vmin=0.5, vmax=1)
        plt.title(title)

        cb = plt.colorbar()
        cb.ax.set_yticklabels(['< 50%', '60%', '70%', '80%', '90%', '100%'])
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation='vertical')
        plt.yticks(tick_marks, classes)

        plt.tight_layout()
        plt.ylabel('classes reais')
        plt.xlabel('classes preditas')
        plt.show()

    @staticmethod
    def plot_feature_importance(vec):
        features = ['Eccentricity', 'Aspect Ratio', 'Elongation', 'Solidity',
                    'Stochastic Convexity', 'Isoperimetric Factor', 'Maximal Indentation Depth',
                    'Lobedness', 'Average Intensity', 'Average Contrast', 'Smoothness',
                    'Third moment', 'Uniformity', 'Entropy']

        l = []
        vec = vec.tolist()
        for i in range(14):
            l.append((vec[i], features[i]))

        l = sorted(l, key=lambda x: x[0], reverse=True)        
        print(l)

        plt.figure()
        plt.title("Importância das características")
        plt.bar(np.arange(14), [x[0] for x in l])
        plt.xticks(np.arange(14), [x[1] for x in l], rotation=45, rotation_mode="anchor", ha="right")
        plt.tight_layout()
        plt.show()