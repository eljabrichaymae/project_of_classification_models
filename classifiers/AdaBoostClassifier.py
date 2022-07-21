from sklearn.tree import DecisionTreeClassifier
from sklearn import ensemble
from classifiers.classifier import Classifier

class AdaBoostClassifier(Classifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = ensemble.AdaBoostClassifier()
        self.param_grid = {
            'base_estimator': [DecisionTreeClassifier(max_depth=10), DecisionTreeClassifier(max_depth=20), DecisionTreeClassifier(max_depth=30)],
            'n_estimators':range(100, 350, 50),
            'learning_rate':[0.25, 0.5, 0.75, 1]
            }
