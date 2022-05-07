from sklearn.neural_network import MLPClassifier
from sklearn import datasets
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import warnings

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
warnings.filterwarnings('ignore')

# pobieranie danych w formacie csv
iris = datasets.load_iris()

mlp = MLPClassifier(activation='tanh', learning_rate='constant')
parameters = {
    'hidden_layer_sizes': [20,40,60,80,100],
    'solver': ['adam'],
    'learning_rate_init': [0.1, 0.01, 0.001]
}
mlp.out_activation_ = 'softmax'
clf = GridSearchCV(mlp, parameters)


# zbiór treningowy
iris_tr = np.concatenate([iris.data[:40], iris.data[50:90], iris.data[100:140]])
iris_tr_target = np.concatenate([iris.target[:40], iris.target[50:90], iris.target[100:140]])

# zbiór testowy
iris_test = np.concatenate([iris.data[40:50], iris.data[90:100], iris.data[140:150]])
iris_test_real = np.concatenate([iris.target[40:50], iris.target[90:100], iris.target[140:150]])

# trenowanie
clf.fit(iris_tr, iris_tr_target)
print(clf.get_params())
#print(clf.cv_results_)

# przewidywanie
iris_pred = clf.predict(iris_test)

#macierz pomyłek
cm = confusion_matrix(iris_pred, iris_test_real)
print("Prediction output: \n", iris_pred)
print("Confusion matrix: \n", cm)

np.set_printoptions(suppress=True)  # nie chcemy naukowej notacji, tylko float ładny
print("Prediction probability for test set: \n", clf.predict_proba(iris_test)*100)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()
