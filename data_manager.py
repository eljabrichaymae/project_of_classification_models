import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS


class DataManager:
    train_file = "data/train.csv"
    classes = []
    encode_classes = []

    def __init__(self, nb_test=0.2, reductionDim=0):
        self.nb_test = nb_test
        self.reductionDim = reductionDim

    def extract_data(self):
        """
        extract data from train_file to obtain training data and validation data
        contains commented code that analyse the initial data set
        :return: x_train, x_test, t_train, t_test
        """
        data = pd.read_csv(self.train_file)
        # print("Le jeu de donnees a pour dimension : ",data.shape)
        # print("Le nom des colonnes est : ", data.head())

        # On voit que l jeu de donnees a 1 id; 1 species; 64 margin; 64 shape; 64 texture
        # print(data["species"].value_counts())
        # En analysant les donnees nous voyons que chaque type de feuilles est represente exactement 10x dans le jeu de donnees

        # verif qu il n y a pas de classe étant null
        '''
        nullClass = data['species'].isnull()
        print(len(nullClass[nullClass == True]))
        '''

        # verification qu'il n'y a pas de cellule NaN
        '''
        data_int_only = data.drop(['id', 'species'], axis=1)
        npdata_int_only = np.array(data_int_only)
        for row in npdata_int_only:
            for cell in row:
                if np.isnan(cell):
                    print(row)
        '''

        # on elimine les donnes qui ne sont pas des int
        data_int_only = data.drop(['id', 'species'], axis=1)
        diff = data_int_only.max() - data_int_only.min()
        diffOverOne = diff[diff >= 1]
        # print("Nombre de colonne ou la difference de valeur est plus grande que 1: ",len(diffOverOne))
        npdata_int_only = np.array(data_int_only)

        if self.reductionDim == 1:
            # Nous appliquons la dimension de reduction pour eviter le curse of dimensionality
            # PCA
            # whiten et random_state font que les donnees seront normaliser en sortie
            pca = PCA(n_components=0.87, whiten=True, random_state=0).fit(npdata_int_only)
            reducedDim = pca.transform(npdata_int_only)

        # print("Dimensions du jeu de données après réduction: {}".format(reducedDim.shape))
        # print("\nCi-dessous nous avons ratio de la variance aux Composants Principaux qui forment 95% de la variance :")
        # print(pca.explained_variance_ratio_)
        # Le jeu de donnees contient maintenant 48 attributs au lieu de 194

        # Reduction avec MDS
        elif self.reductionDim == 2:
            em = MDS(n_components=30)
            reducedDim = em.fit_transform(npdata_int_only)

        # Reduction LLE
        elif self.reductionDim == 3:
            em = LocallyLinearEmbedding(n_components=30)
            reducedDim = em.fit_transform(npdata_int_only)

        # Reduction ISOMAP
        elif self.reductionDim == 4:
            em = Isomap(n_components=30)
            reducedDim = em.fit_transform(npdata_int_only)

        # Cas ou nous ne réduisons pas les donnees
        elif self.reductionDim == 0:
            reducedDim = npdata_int_only

        # Encodage OrdinalEncoder des classes de feuilles
        ordinal_encoder = OrdinalEncoder()
        leafCatEncoder = ordinal_encoder.fit_transform(data[['species']])
        self.classes = list(ordinal_encoder.categories_)[0]
        self.encode_classes = ordinal_encoder.fit_transform(self.classes.reshape(-1, 1))
        x_train, x_test, t_train, t_test = train_test_split(reducedDim, leafCatEncoder,
                                                            test_size=self.nb_test, shuffle=True)
        return x_train, x_test, t_train, t_test
