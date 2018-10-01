import pandas as pd
import numpy as np
from sklearn import datasets


class ReadFiles(object):
    def __init__(self, best=False):
        self.data = pd.read_csv("../files/leaf.csv", header=None)
        self.labels = self.data.pop(0).values
        self.data.pop(1)

        if best:
            aux = self.data[self.data.columns[8]]
            self.data = self.data[self.data.columns[2:6]].join(aux)
            print(self.data)

        self.images = self.data.values

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels
