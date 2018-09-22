from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



class LearnAlgorithms(object):

    def __init__(self, images, labels, cv):
        self.images = images
        self.labels = labels
        self.cv = cv

    def runRFPredict(self):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.images,self.labels)
        y_pred = cross_val_predict(rf, self.images, self.labels, cv=self.cv)
        return confusion_matrix(self.labels, y_pred)

    def runRFScore(self):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.images,self.labels)
        score = cross_val_score(rf, self.images, self.labels, cv=self.cv)
        self.log(score)

    def log(self, msg):
        print('[Learn] {}'.format(msg))

# feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# feature_imp
