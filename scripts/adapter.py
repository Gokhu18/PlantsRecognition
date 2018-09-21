import readfiles
import learnAlgorithms as learn

class Adapter(object):

    def __init__(self, cv):
        rf = readfiles.ReadFiles()
        self.cv = cv
        self.images = rf.getImages()
        self.labels = rf.getLabels()
        self.la = learn.LearnAlgorithms(self.images, self.labels, self.cv)

    def run(self):
        self.la.runRandomForest()

    def log(self, msg):
        print('[Adapter] {}'.format(msg))
