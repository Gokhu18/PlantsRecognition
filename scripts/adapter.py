import readfiles
import learnAlgorithms as learn
from plot import Plot

class Adapter(object):

    def __init__(self, cv, plot):
        rf = readfiles.ReadFiles()
        self.cv = cv
        self.images = rf.getImages()
        self.labels = rf.getLabels()
        self.la = learn.LearnAlgorithms(self.images, self.labels, self.cv)
        self.plot = plot

    def run(self):
        if self.plot == True:
            cof_mat = self.la.runRFPredict()
            namesVec = [0,0,0]
            Plot.plot_confusion_matrix(cof_mat, namesVec, title="Matriz de confusão LDA", normalize=True)
        else:
            self.la.runRFScore()


    def log(self, msg):
        print('[Adapter] {}'.format(msg))
