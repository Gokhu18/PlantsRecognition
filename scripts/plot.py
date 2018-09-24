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

        # for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        #     if cm[i, j] > 0.5: 
        #         if normalize:
        #             string = "{0:.0f}%".format(cm[i, j]*100)
        #         else:
        #             string = "{}".format(cm[i, j])


        #         plt.text(j, i, string,
        #                  horizontalalignment="center",
        #                  color="white" if cm[i, j] > 0.9 else "black")

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

        plt.figure()
        plt.title("Importância das características")
        plt.bar(np.arange(14), vec)
        plt.xticks(np.arange(14), features, rotation=45, rotation_mode="anchor", ha="right")
        plt.tight_layout()
        plt.show()