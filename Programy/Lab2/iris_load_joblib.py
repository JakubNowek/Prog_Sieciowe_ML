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

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
warnings.filterwarnings('ignore')

# pobieranie danych w formacie csv
iris = datasets.load_iris()

# zbiór treningowy
iris_tr = np.concatenate([iris.data[:40], iris.data[50:90], iris.data[100:140]])
iris_tr_target = np.concatenate([iris.target[:40], iris.target[50:90], iris.target[100:140]])

# zbiór testowy
iris_test = np.concatenate([iris.data[40:50], iris.data[90:100], iris.data[140:150]])
iris_test_real = np.concatenate([iris.target[40:50], iris.target[90:100], iris.target[140:150]])

clf = load('modele/irysmodeltest.joblib')
# wypisywanie najlepszych znalezionych parametrów
print("Parametry: \n",clf.get_params())
# ustawienie widoczności całego DataFrame'a
cvResultsDF = pd.DataFrame(clf.cv_results_)
pd.set_option("display.max_rows", None,"display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
# cvresult wypisuje wszystkie możliwości, więc będzie tyle wierszy ile możliwości, czyli tutaj
print("Grid search results: \n", cvResultsDF)

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
