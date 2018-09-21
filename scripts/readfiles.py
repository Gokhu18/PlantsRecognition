import pandas as pd
import numpy as np
from sklearn import datasets


class ReadFiles(object):
    def __init__(self):
        self.data = pd.read_csv("../files/leaf.csv", header=None)
        self.labels = self.data.pop(0).values
        self.images = self.data.values

    def getImages(self):
        return self.images

    def getLabels(self):
        return self.labels
