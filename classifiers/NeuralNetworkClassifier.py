from classifiers.classifier import Classifier
from sklearn.neural_network import MLPClassifier


class NeuralNetworkClassifier(Classifier):
    def __init__(self, data_manager):
        super().__init__(data_manager)
        self.parametric = True
        self.clf = MLPClassifier(max_iter=500)
        self.param_grid = {
            'hidden_layer_sizes': [(100, 100), (100, 100, 100), (100,)],
            'activation': ['tanh', 'relu', 'logistic'],
            'solver': ['lbfgs', 'adam'],
            'alpha': [0.0001, 0.001, 0.01, 0.1, 0.5, 1.0],
            'learning_rate': ['constant', 'adaptive'],
        }

