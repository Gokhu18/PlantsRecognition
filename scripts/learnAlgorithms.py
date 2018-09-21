from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



class LearnAlgorithms(object):

    def __init__(self, images, labels, cv):
        self.images = images
        self.labels = labels
        self.cv = cv

    def runRandomForest(self):
        rf = RandomForestClassifier(n_estimators=100)
        rf.fit(self.images,self.labels)
        scores = cross_val_score(rf, self.images, self.labels, cv=self.cv)
        print(scores)

    def log(self, msg):
        print('[Learn] {}'.format(msg))

#

# RandomForestClassifier(bootstrap=True, class_weight=None, criterion='gini',
#             max_depth=None, max_features='auto', max_leaf_nodes=None,
#             min_impurity_decrease=0.0, min_impurity_split=None,
#             min_samples_leaf=1, min_samples_split=2,
#             min_weight_fraction_leaf=0.0, n_estimators=100, n_jobs=1,
#             oob_score=False, random_state=None, verbose=0,
#             warm_start=False)

# feature_imp = pd.Series(clf.feature_importances_,index=iris.feature_names).sort_values(ascending=False)
# feature_imp
