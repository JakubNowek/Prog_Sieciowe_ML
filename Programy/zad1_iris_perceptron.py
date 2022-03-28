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


# ----------------------part 2--------------------------#

# pobieranie danych w formacie csv
df = pd.read_csv('iris.data', header=None)

df_set = df.iloc[:50]
# print(df_set)
df_ver = df.iloc[50:100]
print(df_ver)
df_vir = df.iloc[100:150]
# print(df_vir)
# print(len(df_set.index), " ", len(df_ver.index), " ", len(df_vir.index))  # sprawdzanie długości data frame

# setosa and versicolor
y = df.iloc[0:150, 4].values  # 100 elementów z 4 kolumny (numeracja od 0) czyli kolumny z nazwą
y = np.where(y == 'Iris-setosa', -1, 1)  # jeśli 'Iris-setosa' zwróć -1, jeśli nie daj 1
# sepal length and petal length
X = df.iloc[0:150, [0, 2]].values  # pobieranie długości kielicha i płatka (kolumny 0 i 2)


# -----------------------part 3--------------------------#
ppn = Perceptron(epochs=10, eta=0.1)  # tworzenie nowego perceptronu

ppn.train(X, y)  # do funkcji train wrzucamy długości i właściwe odpowiedzi
print('Weights: %s' % ppn.w_)
plot_decision_regions(X, y, clf=ppn)  # plotowanie danych z klasyfikacją
plt.title('Perceptron')
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.show()

plt.plot(range(1, len(ppn.errors_) + 1), ppn.errors_, marker='o')  # plotowanie wartości błędów w zależności od iteracji
plt.xlabel('Iterations')
plt.ylabel('Misclassifications')
plt.show()

