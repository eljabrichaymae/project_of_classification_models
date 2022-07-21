from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import numpy as np
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss


class Classifier:
    def __init__(self, data_manager):
        self.dm = data_manager
        self.parametric = False
        self.clf = None
        self.k = 3
        self.x_train, self.x_test, self.t_train, self.t_test = data_manager.extract_data()
        self.training_sample_sizes = np.linspace(0.1, 1.0, 10)

    def train(self):
        """
        choose the best model and train it on training data
        """
        if self.parametric:
            grid_search = GridSearchCV(self.clf, self.param_grid, cv=self.k, refit=True, verbose=True)
            self.clf = grid_search.fit(self.x_train, self.t_train.ravel()).best_estimator_
            print("The best parameters are {} with a score of {:.2f}"
                  .format(grid_search.best_params_, grid_search.best_score_))
        self.clf.fit(self.x_train, self.t_train.ravel())

    def prediction(self):
        """
        :return: prediction of test data
        """
        return self.clf.predict(self.x_test)

    def prediction_prob(self):
        """
        :return: prediction probability of test data
        """
        return self.clf.predict_proba(self.x_test)

    def accuracy(self):
        """
        :return: accuracy of prediction of test data
        """
        return accuracy_score(self.t_test.ravel(), self.prediction())

    def recall(self):
        """
        :return: recall of prediction of test data
        """
        return recall_score(self.t_test.ravel(), self.prediction(), average='macro', zero_division=1)

    def precision(self):
        """
        :return: precision of prediction of test data
        """
        return precision_score(self.t_test.ravel(), self.prediction(), average='macro', zero_division=1)

    def f1score(self):
        """
        :return: f1score of prediction of test data
        """
        return f1_score(self.t_test.ravel(), self.prediction(), average='macro', zero_division=1)

    def report(self):
        """
        :return: report of prediction of test data
        """
        return classification_report(self.t_test.ravel(), self.prediction())

    def loss(self):
        """
        :return: loss of prediction of test data
        """
        return log_loss(self.t_test.ravel(), self.prediction_prob(),labels=self.dm.encode_classes)

    def show_learning_curve(self):
        """
        plot the learning curve
        """
        training_size, training_score, testing_score = learning_curve(estimator=self.clf,X=self.x_train,y=self.t_train.ravel(),train_sizes=self.training_sample_sizes,cv=self.k,n_jobs=-1)
        training_mean = np.mean(training_score, axis=1)
        training_std_deviation = np.std(training_score, axis=1)
        testing_std_deviation = np.std(testing_score, axis=1)
        testing_mean = np.mean(testing_score, axis=1)
        plt.plot(training_size, training_mean, label="Training Data", marker='+', color='blue', markersize=8)
        plt.fill_between(training_size, training_mean + training_std_deviation, training_mean - training_std_deviation,
                         color='blue', alpha=0.12)

        plt.plot(training_size, testing_mean, label="Validation Data", marker='*', color='green', markersize=8)
        plt.fill_between(training_size, testing_mean + training_std_deviation, testing_mean - training_std_deviation,
                         color='green', alpha=0.14)

        plt.title("Scoring of our training and validation data vs sample sizes")
        plt.xlabel("Number of Samples")
        plt.ylabel("Accuracy")
        plt.legend(loc='best')
        plt.show()
