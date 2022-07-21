from classifiers.classifier import Classifier
from sklearn.neighbors import KNeighborsClassifier


class knnClassifier(Classifier):

    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = KNeighborsClassifier()
        self.param_grid = {
            'n_neighbors': range(1, 32, 2),
            'p': [1, 2, 3, 4, 5, 10, 20, 50],
            'metric': ["minkowski", "chebyshev"]
        }
