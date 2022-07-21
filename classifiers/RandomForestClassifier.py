
from classifiers.classifier import Classifier
from sklearn import ensemble


class RandomForestClassifier(Classifier) :
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = ensemble.RandomForestClassifier()
        self.param_grid = {
            'n_estimators': [200, 250, 300, 350],
            'max_depth':[20,30, 40, 50, 60],
            'min_samples_split':[2,4,6],
            'n_jobs': [-1],
            'warm_start': [True],
            }
