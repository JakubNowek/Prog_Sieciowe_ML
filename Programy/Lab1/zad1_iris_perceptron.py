import matplotlib.figure
import matplotlib.pyplot
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

# setosa vs versicolor and virginica

y_training = df_training.iloc[0:120, 4].values  # 100 elementów z 4 kolumny (numeracja od 0) czyli kolumny z nazwą
y_training = np.where(y_training == 'Iris-versicolor', -1, 1)  # jeśli 'Iris-setosa' zwróć -1, jeśli nie daj 1

# sepal length and petal length
 # pobieranie długości kielicha i płatka (kolumny 0 i 2)
X_training = df_training.iloc[0:120, [0, 2]].values
X_test = df_test.iloc[0:30, [0, 2]].values

ppn = Perceptron(epochs=10, eta=0.01)  # tworzenie nowego perceptronu
ppn.train(X_training, y_training)


plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.4)
plt.subplot(1,2,1)
print('Weights: %s' % ppn.w_)
plot_decision_regions(X_training, y_training, clf=ppn)
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
# plt.show()

plt.subplot(1,2,2)
plt.plot(range(1, len(ppn.errors_)+1), ppn.errors_, marker='o')
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')

tytul = "Model perceptronu dla: epochs = {0}, eta = {1}". format(ppn.epochs, ppn.eta)
plt.suptitle(tytul)
plt.show()

# testowanie nauczonego perceptronu
print(ppn.predict(X_test))
