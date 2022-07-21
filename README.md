## Projet-IFT712

Session project which aims to apply and compare different classifiers on a data set

## Running

```console
$ python main.py <classifier> <ratio_test> <dim_reduction_type>

classifier :all|SvmClassifier|knnClassifier| LogisticRegClassifier| NeuralNetworkClassifier|AdaBoostClassifier| RandomForestClassifier
ratio_test : values between 0.2 and 0.3
dim_reduction_type : 1 (PCA)|  2 (Manifold)| 3(LLE) | 4(ISOMAP)| 0 (no reduction)

```
## Dataset
Find the dataset on [Kaggle](https://www.kaggle.com/c/leaf-classification).

## Coding standards
The code respects the [PEP 8](https://www.python.org/dev/peps/pep-0008/) recommandations

