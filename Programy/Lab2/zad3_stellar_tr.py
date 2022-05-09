#fedesoriano. (January 2022). Stellar Classification Dataset - SDSS17.
# Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/stellar-classification-dataset-sdss17.
# Classification of Stars, Galaxies and Quasars. Sloan Digital Sky Survey DR17

import tensorflow
from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import warnings
from joblib import dump, load

pd.set_option("display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
np.set_printoptions(linewidth=150)

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
warnings.filterwarnings('ignore')


# wczytywanie csv
stellar_df = pd.read_csv('star_classification.csv')

stellar_df = stellar_df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift']]
# print(stellar_df.head())
print("Wczytano bazę.\n")

# podział zestawu na klasy
galaxy = stellar_df[stellar_df['class'] == 'GALAXY']
star = stellar_df[stellar_df['class'] == 'STAR']
qso = stellar_df[stellar_df['class'] == 'QSO']

# galaxy = galaxy.reset_index(drop=True)
# star = star.reset_index(drop=True)
# qso = qso.reset_index(drop=True)

galaxy = galaxy.sample(10100).drop(labels='class', axis='columns')
star = star.sample(10100).drop(labels='class', axis='columns')
qso = qso.sample(10100).drop(labels='class', axis='columns')


# wyjścia zbioru treningowego
galaxy_target = np.array([0]*10100)
star_target = np.array([1]*10100)
qso_target = np.array([2]*10100)

# zbiór treningowy
stel_tr = np.concatenate([galaxy[:10000], star[:10000], qso[:10000]])
stel_tr_target = np.concatenate((galaxy_target[:10000], star_target[:10000], qso_target[:10000]))
# print(stel_tr.shape)
# print(stel_tr_target.shape)

# zbiór testowy
stel_test = np.concatenate([galaxy[10000:10100], star[10000:10100], qso[10000:10100]])
stel_test_real = np.concatenate((galaxy_target[10000:10100], star_target[10000:10100], qso_target[10000:10100]))

# tworzenie wielowarstwowego perceptronu
mlp = MLPClassifier(activation='tanh', learning_rate='constant', max_iter=50)
parameters = {
    'hidden_layer_sizes': [60,80,100],
    'solver': ['adam'],
    'learning_rate_init': [0.1, 0.01, 0.001]
}
mlp.out_activation_ = 'softmax'
clf = GridSearchCV(mlp, parameters)
clf.out_activation_ = 'softmax'
# trenowanie
clf.fit(stel_tr, stel_tr_target)
dump(clf, 'stellar_mod_soft.joblib')
