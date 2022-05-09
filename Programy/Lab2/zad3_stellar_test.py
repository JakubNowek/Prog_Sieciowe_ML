import tensorflow
from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load

pd.set_option("display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
np.set_printoptions(linewidth=100)

# wczytywanie csv
stellar_df = pd.read_csv('star_classification.csv')

# wybranie tylko istotnych kolumn
stellar_df = stellar_df[['alpha', 'delta', 'u', 'g', 'r', 'i', 'z', 'class', 'redshift']]
print("Wczytano bazę.\n")

# podział zestawu na klasy
galaxy = stellar_df[stellar_df['class'] == 'GALAXY']
star = stellar_df[stellar_df['class'] == 'STAR']
qso = stellar_df[stellar_df['class'] == 'QSO']


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
clf = load('modele/stellar_mod_sgd.joblib')
# wypisywanie najlepszych znalezionych parametrów
modelParams = pd.Series(clf.get_params())
print("Parametry: \n", modelParams)
# ustawienie widoczności całego DataFrame'a
cvResultsDF = pd.DataFrame(clf.cv_results_)
pd.set_option("display.max_rows", None,"display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
np.set_printoptions(linewidth=None)
# cvresult wypisuje wszystkie możliwości, więc będzie tyle wierszy ile możliwości, czyli tutaj
#print("Grid search results: \n", cvResultsDF)
print("Grid search results: \n", cvResultsDF[["params", "mean_test_score","rank_test_score"]])
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