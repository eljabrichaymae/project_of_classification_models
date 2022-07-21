from data_manager import DataManager
from classifiers.SvmClassifier import SvmClassifier
from classifiers.knnClassifier import knnClassifier
from classifiers.LogisticRegClassifier import LogisticRegClassifier
from classifiers.NeuralNetworkClassifier import NeuralNetworkClassifier
from classifiers.RandomForestClassifier import RandomForestClassifier
from classifiers.AdaBoostClassifier import AdaBoostClassifier
import sys


def main():
    usage = " Usage : python main.py classifier ratio_test dim_reduction_type\
            \n\t classifier can take the following values :\
            \n\t SvmClassifier or knnClassifier or LogisticRegClassifier or NeuralNetworkClassifier or RandomForestClassifier or AdaBoostClassifier\
            \n\t can also take as value all  \
            \n\t ratio_test : determine the ration of test data from the original dataset (0.2<ratio_test<0.3) \
            \n\t dim_reduction_type : Type of dimensionality reduction : { 1: PCA, 2: Manifold, 3: LLE, 4: ISOMAP, 0: no reduction}  \
            \n"

    print(usage)
    classifier = sys.argv[1]
    rt_test = float(sys.argv[2])
    dim_reduction_type = int(sys.argv[3])

    dm = DataManager(nb_test=rt_test, reductionDim=dim_reduction_type )
    if classifier != "all":
        classifier = getattr(sys.modules[__name__], classifier)
        clf = classifier(dm)
        clf.train()
        print('Accuracy: ' + str(clf.accuracy()))
        print('Precision: ' + str(clf.precision()))
        print('Recall: ' + str(clf.recall()))
        print('F1-score: ' + str(clf.f1score()))
        print('Loss: ' + str(clf.loss()))
        clf.show_learning_curve()
    else:
        classifiers = [SvmClassifier, knnClassifier, LogisticRegClassifier, NeuralNetworkClassifier, AdaBoostClassifier,
                       RandomForestClassifier]
        for classifier in classifiers:
            clf = classifier(dm)
            clf.train()
            print('Accuracy: ' + str(clf.accuracy()))
            print('Precision: ' + str(clf.precision()))
            print('Recall: ' + str(clf.recall()))
            print('F1-score: ' + str(clf.f1score()))
            print('Loss: ' + str(clf.loss()))


if __name__ == "__main__":
    main()
