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


# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
# warnings.filterwarnings('ignore')

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

clf = load('mnist_mod.joblib')
# wypisywanie najlepszych znalezionych parametrów
print("Parametry: \n",clf.get_params())
# ustawienie widoczności całego DataFrame'a
cvResultsDF = pd.DataFrame(clf.cv_results_)
pd.set_option("display.max_rows", None,"display.max_columns", None,
              "max_colwidth", None, "display.expand_frame_repr", False)
# cvresult wypisuje wszystkie możliwości, więc będzie tyle wierszy ile możliwości, czyli tutaj
print("Grid search results: \n", cvResultsDF)


mnist_pred = clf.predict(test_X[:10])
#macierz pomyłek
cm = confusion_matrix(mnist_pred, test_y[:10])
print("Prediction output: \n", mnist_pred)
print("Confusion matrix: \n", cm)

np.set_printoptions(suppress=True)  # nie chcemy naukowej notacji, tylko float ładny
#print("Prediction probability for test set: \n", clf.predict_proba(mnist_pred)*100)
plt.matshow(cm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# print(to_image(test_X[0]))

for i in range(9):
    plt.subplot(330 + 1 + i)
    plt.imshow(to_image(test_X[i]), cmap=plt.get_cmap('gray'))
plt.show()