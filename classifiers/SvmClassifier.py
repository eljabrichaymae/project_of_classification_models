from classifiers.classifier import Classifier
from sklearn.svm import SVC


class SvmClassifier(Classifier):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = SVC()
        self.param_grid = {
            "kernel": ['rbf', 'sigmoid', 'poly'],
            "C": [0.000001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            "gamma": [0.000001, 0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 'scale', 'auto'],
            "coef0": [15, 16, 17, 18, 19, 20]
        }

    def train(self):
        """
        override of train method to select the best model before setting probability to True,
        in purpose to accelerate the GridSearchCv
        """
        super().train()
        self.clf.probability = True
        self.clf.fit(self.x_train, self.t_train)
