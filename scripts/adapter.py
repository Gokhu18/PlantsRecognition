import readfiles
import learnAlgorithms as learn
from plot import Plot
import numpy as np

class Adapter(object):

    def __init__(self, cv, plot, number_of_trees, best):
        self.best = best
        rf = readfiles.ReadFiles(best=self.best)
        self.cv = cv
        self.number_of_trees = number_of_trees
        self.images = rf.getImages()
        self.labels = rf.getLabels()
        self.la = learn.LearnAlgorithms(self.images, self.labels, self.cv, self.number_of_trees)
        self.plot = plot

    def run(self):
        if self.plot == True:
            feature_importance, cof_mat = self.la.runRFPredict()
            namesVec = np.arange(1,31)
            print(cof_mat)
            Plot.plot_confusion_matrix(cof_mat, namesVec, title="Matriz de confusão LDA", normalize=True)
            Plot.plot_feature_importance(feature_importance)
        else:
            self.la.runRFScore()


    def log(self, msg):
        print('[Adapter] {}'.format(msg))
