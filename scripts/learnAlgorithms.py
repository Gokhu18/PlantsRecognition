from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from plot import Plot



class LearnAlgorithms(object):

    def __init__(self, images, labels, cv, number_of_trees = 100):
        self.images = images
        self.labels = labels
        self.cv = cv
        self.number_of_trees = number_of_trees

    def runRFPredict(self):
        rf = RandomForestClassifier(n_estimators=self.number_of_trees)
        rf.fit(self.images, self.labels)
        y_pred = cross_val_predict(rf, self.images, self.labels, cv=self.cv)
        return rf.feature_importances_, confusion_matrix(self.labels, y_pred)

    def runRFScore(self):
        rf = RandomForestClassifier(n_estimators=self.number_of_trees)
        rf.fit(self.images,self.labels)
        score = cross_val_score(rf, self.images, self.labels, cv=self.cv)
        self.log(score)

    def log(self, msg):
        print('[Learn] {}'.format(msg))
