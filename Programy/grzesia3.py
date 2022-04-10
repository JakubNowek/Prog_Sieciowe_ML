import numpy as np


# class AdalineGD(object):

#    def __init__(self, eta=0.01, epochs=50):
#        self.eta = eta
#        self.epochs = epochs

#    def train(self, X, y):

#        self.w_ = np.zeros(1 + X.shape[1])
#        self.cost_ = []

#        for i in range(self.epochs):
#            output = self.net_input(X)
#            errors = (y - output)
#            self.w_[1:] += self.eta * X.T.dot(errors)
#            self.w_[0] += self.eta * errors.sum()
#            cost = (errors**2).sum() / 2.0
#            self.cost_.append(cost)
#        return self

#    def net_input(self, X):
#        return np.dot(X, self.w_[1:]) + self.w_[0]

#    def activation(self, X):
#        return self.net_input(X)

#    def predict(self, X):
#        return np.where(self.activation(X) >= 0., 1, -1)

class AdalineSGD(object):

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


import pandas as pd

df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data', header=None)

X = df.iloc[0:40, [0, 2]].values
Y = df.iloc[50:90, [0, 2]].values
Z = df.iloc[100:140, [0, 2]].values
Test1 = np.vstack((X, Y))  # setosa and versicolor
Test2 = np.vstack((Y, Z))  # versicolor and virginica
Test3 = np.vstack((X, Z))  # setosa and virginica

a = df.iloc[0:40, 4].values
b = df.iloc[50:90, 4].values
c = df.iloc[100:140, 4].values
Test_v1 = np.concatenate([a, b], axis=None)  # setosa and versicolor
Test_v2 = np.concatenate([b, c], axis=None)  # versicolor and virginica
Test_v3 = np.concatenate([a, c], axis=None)  # setosa and virginica

Class = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

print(Class[0])
print(Class[1])
print(Class[2])

Test_versicolor_v1 = np.where(Test_v1 == 'Iris-setosa', -1, 1)
Test_versicolor_v2 = np.where(Test_v2 == 'Iris-setosa', -1, 1)
Test_versicolor_v3 = np.where(Test_v3 == 'Iris-setosa', -1, 1)

Setosa_data = df.iloc[40:50, [0, 2]].values
Versicolor_data = df.iloc[90:100, [0, 2]].values
Virginica_data = df.iloc[140:150, [0, 2]].values

import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions

XY_std = np.copy(Test1)
XY_std[:, 0] = (Test1[:, 0] - Test1[:, 0].mean()) / Test1[:, 0].std()
XY_std[:, 1] = (Test1[:, 1] - Test1[:, 1].mean()) / Test1[:, 1].std()

np.random.seed(123)
idx = np.random.permutation(len(Test_versicolor_v1))
XY_shuffled, Test1_shuffled = XY_std[idx], Test_versicolor_v1[idx]

YZ_std = np.copy(Test2)
YZ_std[:, 0] = (Test2[:, 0] - Test2[:, 0].mean()) / Test2[:, 0].std()
YZ_std[:, 1] = (Test2[:, 1] - Test2[:, 1].mean()) / Test2[:, 1].std()

np.random.seed(123)
idx = np.random.permutation(len(Test_versicolor_v2))
YZ_shuffled, Test2_shuffled = YZ_std[idx], Test_versicolor_v2[idx]

XZ_std = np.copy(Test3)
XZ_std[:, 0] = (Test3[:, 0] - Test3[:, 0].mean()) / Test3[:, 0].std()
XZ_std[:, 1] = (Test3[:, 1] - Test3[:, 1].mean()) / Test3[:, 1].std()

np.random.seed(123)
idx = np.random.permutation(len(Test_versicolor_v3))
XZ_shuffled, Test3_shuffled = XZ_std[idx], Test_versicolor_v3[idx]

Setosa_std = np.copy(Setosa_data)
Setosa_std[:, 0] = (Setosa_data[:, 0] - Setosa_data[:, 0].mean()) / Setosa_data[:, 0].std()
Setosa_std[:, 1] = (Setosa_data[:, 1] - Setosa_data[:, 1].mean()) / Setosa_data[:, 1].std()

print(Setosa_std)
Versicolor_std = np.copy(Versicolor_data)
Versicolor_std[:, 0] = (Versicolor_data[:, 0] - Versicolor_data[:, 0].mean()) / Versicolor_data[:, 0].std()
Versicolor_std[:, 1] = (Versicolor_data[:, 1] - Versicolor_data[:, 1].mean()) / Versicolor_data[:, 1].std()

Virginica_std = np.copy(Virginica_data)
Virginica_std[:, 0] = (Virginica_data[:, 0] - Virginica_data[:, 0].mean()) / Virginica_data[:, 0].std()
Virginica_std[:, 1] = (Virginica_data[:, 1] - Virginica_data[:, 1].mean()) / Virginica_data[:, 1].std()

Set_Ver = AdalineSGD(epochs=50, eta=0.00001)
Ver_Vir = AdalineSGD(epochs=10, eta=0.01)
Set_Vir = AdalineSGD(epochs=50, eta=0.00001)

Set_Ver.train(XY_shuffled, Test1_shuffled)

# plot_decision_regions(XY_shuffled, Test1_shuffled, clf=Set_Ver)
# plt.title('Adaline - Gradient Descent Setosa v Versicolor')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.show()

Ver_Vir.train(YZ_shuffled, Test2_shuffled)

# plot_decision_regions(YZ_shuffled, Test2_shuffled, clf=Ver_Vir)
# plt.title('Adaline - Gradient Descent Versicolor v Virginica')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.show()

Set_Vir.train(XZ_shuffled, Test3_shuffled)

# plot_decision_regions(XZ_shuffled, Test3_shuffled, clf=Set_Vir)
# plt.title('Adaline - Gradient Descent Setosa v Virginica')
# plt.xlabel('sepal length [standardized]')
# plt.ylabel('petal length [standardized]')
# plt.show()

print(Set_Ver.predict(Setosa_std))
print(Ver_Vir.predict(Setosa_std))
print(Set_Vir.predict(Setosa_std))

