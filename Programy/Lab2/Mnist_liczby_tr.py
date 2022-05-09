# https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python

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

def reshape(set_of_arrays):
    list_of_vectors = []
    for array in set_of_arrays:
        vector = np.reshape(array, -1)
        list_of_vectors.append(vector)
    return np.array(list_of_vectors)

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
warnings.filterwarnings('ignore')

# ładowanie zestawu danych mnist
(train_X, train_y), (test_X, test_y) = mnist.load_data()

# shape of dataset
print('X_train: ' + str(train_X.shape))
print('Y_train: ' + str(train_y.shape))
print('X_test:  ' + str(test_X.shape))
print('Y_test:  ' + str(test_y.shape))

# reshaping np arrays to np vectors
train_X = reshape(train_X)
#Y_train = reshape(train_y)
test_X = reshape(test_X)
#Y_test = reshape(test_y)
print("Reshaping finished:")
print('X_train: ' + str(train_X.shape))
print('X_test:  ' + str(test_X.shape))
# tworzenie wielowarstwowego perceptronu
mlp = MLPClassifier(activation='tanh', learning_rate='constant', max_iter=20)
parameters = {
    'hidden_layer_sizes': [20,40,60,80,100],
    'solver': ['sgd'],
    'learning_rate_init': [0.1, 0.01, 0.001]
}
mlp.out_activation_ = 'softmax'
clf = GridSearchCV(mlp, parameters)

# trenowanie
print("Training...")
clf.fit(train_X, train_y)
# zapisywanie wytrenowanego modelu
dump(clf, 'mnist_jelba.joblib')
