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
# setosa vs (versicolor + virginica)
y_tr_setosa = np.where(y_training == 'Iris-setosa', -1, 1)  # jeśli 'Iris-setosa' zwróć -1, jeśli nie daj 1
# versicolor vs (setosa + virginica)
y_tr_versicolor = np.where(y_training == 'Iris-versicolor', -1, 1)  # jeśli 'Iris-versicolor' zwróć -1, jeśli nie daj 1
# virginica vs (versicolor + setosa)
y_tr_virginica = np.where(y_training == 'Iris-virginica', -1, 1)  # jeśli 'Iris-virginica' zwróć -1, jeśli nie daj 1

# pobieranie długości kielicha i płatka (kolumny 0 i 2)
X_training = df_training.iloc[0:120, [0, 2]].values
X_test = df_test.iloc[0:30, [0, 2]].values


ppn_setosa = Perceptron(epochs=10, eta=0.1)  # tworzenie nowego perceptronu rozpoznającego setosy
ppn_versicolor = Perceptron(epochs=10, eta=0.1)  # tworzenie nowego perceptronu rozpoznającego versicolory
ppn_virginica = Perceptron(epochs=10, eta=0.1)  # tworzenie nowego perceptronu rozpoznającego virginice

ppn.train(X_training, y_training)
ppn.train(X_training, y_training)
ppn.train(X_training, y_training)

print('Weights: %s' % ppn.w_)
plot_decision_regions(X_training, y_training, clf=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

# testowanie nauczonego perceptronu
print(ppn.predict(X_test))
