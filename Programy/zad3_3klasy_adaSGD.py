import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class AdalineSGD(object):  # stochastic gradient descent

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y, reinitialize_weights=True):

        if reinitialize_weights:
            self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            for xi, target in zip(X, y):
                output = self.net_input(xi)
                error = (target - output)
                self.w_[1:] += self.eta * xi.dot(error)
                self.w_[0] += self.eta * error

            cost = ((y - self.activation(X)) ** 2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        return self.net_input(X)

    def predict(self, X):
        return np.where(self.activation(X) >= 0.0, 1, -1)


# pobieranie danych w formacie csv
df = pd.read_csv('iris.data', header=None)

# wydzielenie danych treningowych
df_set = df.iloc[:40]
df_ver = df.iloc[50:90]
df_vir = df.iloc[100:140]
df_training = pd.concat([df_set, df_ver, df_vir], ignore_index=True)  # zbiór treningowy - 40x40x40

# wydzielenie danych testowych
test_df_set = df.iloc[40:50]
test_df_ver = df.iloc[90:100]
test_df_vir = df.iloc[140:150]
df_test = pd.concat((test_df_set, test_df_ver, test_df_vir), ignore_index=True)  # zbiór testowy 10x10x10

y_training = df_training.iloc[0:120, 4].values  # 100 elementów z 4 kolumny (numeracja od 0) czyli kolumny z nazwą
# setosa vs (versicolor + virginica)
y_tr_setosa = np.where(y_training == 'Iris-setosa', -1, 1)  # jeśli 'Iris-setosa' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_setosa)  # debugging
# versicolor vs (setosa + virginica)
y_tr_versicolor = np.where(y_training == 'Iris-versicolor', -1, 1)  # jeśli 'Iris-versicolor' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_versicolor)  # debugging
# virginica vs (versicolor + setosa)
y_tr_virginica = np.where(y_training == 'Iris-virginica', -1, 1)  # jeśli 'Iris-virginica' zwróć -1, jeśli nie daj 1
### print("trening\n", y_tr_virginica)  # debugging

# Tworzenie zbioru treningowego i testowego - pobieranie długości kielicha i płatka (kolumny 0 i 2)
X_training = df_training.iloc[0:120, [0, 2]].values
X_test = df_test.iloc[0:30, [0, 2]].values

# Standaryzowanie danych
# standaryzowanie zbioru uczącego
X_train_std = np.copy(X_training)
X_train_std[:,0] = (X_training[:,0] - X_training[:,0].mean()) / X_training[:,0].std()
X_train_std[:,1] = (X_training[:,1] - X_training[:,1].mean()) / X_training[:,1].std()

# standaryzowanie zbioru treningowego
X_test_std = np.copy(X_test)
X_test_std[:, 0] = (X_test[:, 0] - X_test[:, 0].mean()) / X_test[:, 0].std()
X_test_std[:, 1] = (X_test[:, 1] - X_test[:, 1].mean()) / X_test[:, 1].std()

# tworzenie 3 klasyfikatorów
ada_setosa = AdalineSGD(epochs=50, eta=0.001)
ada_versicolor = AdalineSGD(epochs=50, eta=0.001)
ada_virginica = AdalineSGD(epochs=50, eta=0.001)

# train and adaline and plot decision regions
ada_setosa.train(X_train_std, y_tr_setosa)
ada_versicolor.train(X_train_std, y_tr_versicolor)
ada_virginica.train(X_train_std, y_tr_virginica)

# plot_decision_regions(X_train_std, y_training, clf=ada)
# plt.title('Adaline - Gradient Descent')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.show()

# ada_output = ada.net_input(X_test_std)  # o(x)
# #print(ada_output)
# plt.plot(range(1, len(ada_output)+1), ada_output, marker='o')
# plt.title('Adaline - o(x) dla z ze zboru walidacyjnego')
# plt.xlabel('indeks x ze zbioru walidacyjnego')
# plt.ylabel('o(x)')
# plt.show()

# testowanie
print(ada_setosa.predict(X_test_std))
print(ada_versicolor.predict(X_test_std))
print(ada_virginica.predict(X_test_std))
# plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
# plt.title('Wartość błędu w zależności od ilości iteracji')
# plt.xlabel('Ilość iteracji')
# plt.ylabel('Sum-squared-error')
# plt.show()
