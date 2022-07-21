from classifiers.classifier import Classifier
from sklearn.linear_model import LogisticRegression
import numpy as np


class LogisticRegClassifier(Classifier):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = LogisticRegression()
        self.param_grid = {
            "penalty": ['l1', 'l2'],
            "C": np.logspace(-4, 4, 20),
            "solver": ['liblinear'],
        }
