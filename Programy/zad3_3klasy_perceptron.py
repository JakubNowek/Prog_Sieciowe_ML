import numpy as np
import pandas as pd

#  %matplotlib inline
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class Perceptron(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):

        self.w_ = np.zeros(1 + X.shape[1])
        self.errors_ = []

        for _ in range(self.epochs):
            errors = 0
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
                errors += int(update != 0.0)
            self.errors_.append(errors)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


# pobieranie danych w formacie csv
df = pd.read_csv('iris.data', header=None)

df_set = df.iloc[:40]
# print(df_set)
df_ver = df.iloc[50:90]
# print(df_ver)
df_vir = df.iloc[100:140]
# print(df_vir)

df_training = pd.concat([df_set, df_ver, df_vir], ignore_index=True)  # zbiór treningowy - 40x40x40

test_df_set = df.iloc[40:50]
test_df_ver = df.iloc[90:100]
test_df_vir = df.iloc[140:150]

df_test = pd.concat((test_df_set, test_df_ver, test_df_vir), ignore_index=True)  # zbiór testowy 10x10x10

y_training = df_training.iloc[0:120, 4].values  # 100 elementów z 4 kolumny (numeracja od 0) czyli kolumny z nazwą
### print("trening\n", y_training)  # debugging
# setosa vs (versicolor + virginica)
y_tr_setosa = np.where(y_training == 'Iris-setosa', -1, 1)  # jeśli 'Iris-setosa' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_setosa)  # debugging
# versicolor vs (setosa + virginica)
y_tr_versicolor = np.where(y_training == 'Iris-versicolor', -1, 1)  # jeśli 'Iris-versicolor' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_versicolor)  # debugging
# virginica vs (versicolor + setosa)
y_tr_virginica = np.where(y_training == 'Iris-virginica', -1, 1)  # jeśli 'Iris-virginica' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_virginica)  # debugging

# pobieranie długości kielicha i płatka (kolumny 0 i 2)
X_training = df_training.iloc[0:120, [0, 2]].values
X_test = df_test.iloc[0:30, [0, 2]].values


X_train_std = np.copy(X_training)
X_train_std[:,0] = (X_training[:,0] - X_training[:,0].mean()) / X_training[:,0].std()
X_train_std[:,1] = (X_training[:,1] - X_training[:,1].mean()) / X_training[:,1].std()

# standaryzowanie zbioru treningowego
X_test_std = np.copy(X_test)
X_test_std[:, 0] = (X_test[:, 0] - X_test[:, 0].mean()) / X_test[:, 0].std()
X_test_std[:, 1] = (X_test[:, 1] - X_test[:, 1].mean()) / X_test[:, 1].std()

ppn_setosa = Perceptron(epochs=10, eta=0.01)  # tworzenie nowego perceptronu rozpoznającego setosy
ppn_versicolor = Perceptron(epochs=10, eta=0.001)  # tworzenie nowego perceptronu rozpoznającego versicolory
ppn_virginica = Perceptron(epochs=100, eta=0.01)  # tworzenie nowego perceptronu rozpoznającego virginice

ppn_setosa.train(X_train_std, y_tr_setosa)
ppn_versicolor.train(X_train_std, y_tr_versicolor)
ppn_virginica.train(X_train_std, y_tr_virginica)

# print('Weights: %s' % ppn_setosa.w_)
# print('Weights: %s' % ppn_versicolor.w_)
# print('Weights: %s' % ppn_virginica.w_)

plot_decision_regions(X_train_std, y_tr_setosa, clf=ppn_setosa)
# plt.title('Perceptron')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')

plot_decision_regions(X_train_std, y_tr_versicolor, clf=ppn_versicolor)
# plt.title('Perceptron')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')

plot_decision_regions(X_train_std, y_tr_virginica, clf=ppn_virginica)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()
# plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
# plt.xlabel('Iterations')
# plt.ylabel('Misclassifications')
# plt.show()

# testowanie nauczonego perceptronu
result_set = list(ppn_setosa.predict(X_test_std))
print(result_set)
result_ver = list(ppn_versicolor.predict(X_test_std))
print(result_ver)
result_vir = list(ppn_virginica.predict(X_test_std))
print(result_vir)

# ostateczny wynik algorytmu? jeśli wykryto w jednej to nazwa, a jeśli w dwóch to brak nazwy
result_array = np.array([result_set, result_ver, result_vir])
result_list = []
print(result_array)
array_sum = np.sum(result_array, axis=0)
# result = list(np.zeros(len(X_test_std)))
for i in range(np.shape(result_array)[1]):  # iterowanie po ilości kolumn
    if array_sum[i] == 1:  # sumowanie elementów w kolumnach
        if result_set[i] == -1:
            result_list.append('Set')
        if result_ver[i] == -1:
            result_list.append('Ver')
        if result_vir[i] == -1:
            result_list.append('Vir')
    else:
        result_list.append('-')
print(result_list)

