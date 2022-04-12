import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mlxtend.plotting import plot_decision_regions


class AdalineGD(object):

    def __init__(self, eta=0.01, epochs=50):
        self.eta = eta
        self.epochs = epochs

    def train(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        self.cost_ = []

        for i in range(self.epochs):
            output = self.net_input(X)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors ** 2).sum() / 2.0
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
X_training = df_training.iloc[0:120, [0, 1, 2, 3]].values
X_test = df_test.iloc[0:30, [0, 1, 2, 3]].values

# Standaryzowanie danych
# standaryzowanie zbioru uczącego
X_train_std = np.copy(X_training)
X_train_std[:,0] = (X_training[:,0] - X_training[:,0].mean()) / X_training[:,0].std()
X_train_std[:,1] = (X_training[:,1] - X_training[:,1].mean()) / X_training[:,1].std()
X_train_std[:,2] = (X_training[:,2] - X_training[:,2].mean()) / X_training[:,2].std()
X_train_std[:,3] = (X_training[:,3] - X_training[:,3].mean()) / X_training[:,3].std()
# standaryzowanie zbioru treningowego
X_test_std = np.copy(X_test)
X_test_std[:, 0] = (X_test[:, 0] - X_test[:, 0].mean()) / X_test[:, 0].std()
X_test_std[:, 1] = (X_test[:, 1] - X_test[:, 1].mean()) / X_test[:, 1].std()
X_test_std[:, 2] = (X_test[:, 2] - X_test[:, 2].mean()) / X_test[:, 2].std()
X_test_std[:, 3] = (X_test[:, 3] - X_test[:, 3].mean()) / X_test[:, 3].std()

# tworzenie 3 klasyfikatorów
ada_setosa = AdalineGD(epochs=100, eta=0.01)
ada_versicolor = AdalineGD(epochs=100, eta=0.01)
ada_virginica = AdalineGD(epochs=1000, eta=0.01)

# train and adaline and plot decision regions
ada_setosa.train(X_train_std, y_tr_setosa)
ada_versicolor.train(X_train_std, y_tr_versicolor)
ada_virginica.train(X_train_std, y_tr_virginica)

# plot_decision_regions(X_train_std, y_tr_setosa, clf=ada_setosa)
# plt.title('AdalineGD')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')

# plot_decision_regions(X_train_std, y_tr_versicolor, clf=ada_versicolor)
# plt.title('AdalineGD')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')

# plot_decision_regions(X_train_std, y_tr_virginica, clf=ada_virginica)
# plt.title('AdalineGD')
# plt.xlabel('sepal length [cm]')
# plt.ylabel('petal length [cm]')
#####plt.show()

# ada_output = ada.net_input(X_test_std)  # o(x)
# #print(ada_output)
# plt.plot(range(1, len(ada_output)+1), ada_output, marker='o')
# plt.title('Adaline - o(x) dla z ze zboru walidacyjnego')
# plt.xlabel('indeks x ze zbioru walidacyjnego')
# plt.ylabel('o(x)')
# plt.show()

# testowanie
result_set = ada_setosa.predict(X_test_std)
print(result_set)
result_ver = ada_versicolor.predict(X_test_std)
print(result_ver)
result_vir = ada_virginica.predict(X_test_std)
print(result_vir)
# plt.plot(range(1, len(ada.cost_)+1), ada.cost_, marker='o')
# plt.title('Wartość błędu w zależności od ilości iteracji')
# plt.xlabel('Ilość iteracji')
# plt.ylabel('Sum-squared-error')
# plt.show()
result_list = list(np.zeros(len(X_test_std)))

for i in range(len(X_test_std)):  # iterowanie po ilości kolumn
    isDetected = False
    error = 0
    # setosa
    if result_set[i] == -1:
        result_list[i] = 'Set'
        isDetected = True
        error = ada_setosa.cost_[-1]
    # versicolor
    if result_ver[i] == -1 and isDetected == True:
        if ada_versicolor.cost_[-1]<error:
            result_list[i] = 'Ver'
    if result_ver[i] == -1 and isDetected == False:
        result_list[i] = 'Ver'
        isDetected = True
        error = ada_versicolor.cost_[-1]
    # virginica
    if result_vir[i] == -1 and isDetected == True:
        if ada_virginica.cost_[-1] < error:
            result_list[i] = 'Ver'
    if result_vir[i] == -1 and isDetected == False:
        result_list[i] = 'Vir'
        isDetected = True
# print(result_list)
for i in range(len(result_list)):  # jeśli żaden klasyfikator nie wykrył, wpisz '-'
    if result_list[i] == 0:
        result_list[i] = '-'
# print(result_list)
print("Co przewidział klasyfikator")
print("Epochs Setosa= ", ada_setosa.epochs,
      "/ Epochs Versicolor= ", ada_versicolor.epochs,
      "/ Epochs Virginica= ", ada_virginica.epochs)
print("Eta Setosa= ", ada_setosa.eta,
      "/ Eta Versicolor= ", ada_versicolor.eta,
      "/ Eta Virginica= ", ada_virginica.eta)
print(result_list[:10])
print(result_list[10:20])
print(result_list[20:30])
# oczywiście można by zrobić, że do każdego kwiatu testowego wypisuje, z jakim błędem stwierdza, że to to lub nie to,
# ale nie było to wymagane w zadaniu
