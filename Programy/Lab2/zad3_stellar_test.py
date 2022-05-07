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

# podajemy np.array 1xn i otrzymujemy podzieloną np.array 28x28
def to_image(vector):
    vector = vector.tolist()
    vector = [vector[x:x+28] for x in range(0, len(vector), 28)]
    vector = np.array(vector)
    return vector


pd.set_option("display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
np.set_printoptions(linewidth=150)

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
# warnings.filterwarnings('ignore')


# wczytywanie csv
stellar_df = pd.read_csv('star_classification.csv')

stellar_df = stellar_df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift']]
# print(stellar_df.head())
print("Wczytano bazę.")

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

#ładowanie klasyfikatora
clf = load('stellar_mod.joblib')
# wypisywanie najlepszych znalezionych parametrów
modelParams = pd.Series(clf.get_params())
print("Parametry: \n", modelParams)
# ustawienie widoczności całego DataFrame'a
cvResultsDF = pd.DataFrame(clf.cv_results_)
pd.set_option("display.max_rows", None,"display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
np.set_printoptions(linewidth=None)
# cvresult wypisuje wszystkie możliwości, więc będzie tyle wierszy ile możliwości, czyli tutaj
print("Grid search results: \n", cvResultsDF)

# przewidywanie
iris_pred = clf.predict(stel_test)
#macierz pomyłek
cm = confusion_matrix(iris_pred, stel_test_real)
print("Prediction output: \n", iris_pred)
print("Confusion matrix: \n", cm)

np.set_printoptions(suppress=True)  # nie chcemy naukowej notacji, tylko float ładny
print("Prediction probability for test set: \n", clf.predict_proba(stel_test)*100)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()