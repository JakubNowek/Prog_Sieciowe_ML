from math import sqrt,exp
import tqdm
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from itertools import combinations


class Kohonen:
    # Atrybuty nadawane przy tworzeniu obiektu.
    def __init__(self, p:int, method="cosine", alpha_method="linear"):
        """
        @param p: liczba klas
        @param method: jedna z 3 metod do wyznaczania normy
        @param alpha_method: jedna z 3 metod aktualizacji współczynnika uczenia
        """
        self.n_clusters = p
        self.method = method
        self.alpha_method = alpha_method

    def fit_transform(self, X, iterations, alpha_init):
        """
        @param X: dane
        @param iterations: liczba iteracji
        @param alpha_init: początkowa wartość współczynnika uczenia
        @return: tablica o rozmiarze N, przechowująca dla każdej próbki wartość klastra, do którego należy
        """
        # współczynnik uczenia
        alpha = alpha_init
        # środek ciężkości zbioru
        s = X.sum(axis=0)/len(X)
        # normalizacja danych
        X_norm = np.array([(s - x)/np.linalg.norm(x, axis=-1) for x in X])
        # inicjowanie wektorów w (wektor w jest centroidą)
        w = np.random.random((self.n_clusters, len(X[0])))
        w = w / np.linalg.norm(w)

        for k in tqdm.tqdm(range(iterations)):
            # wyznaczanie miary i przypisywanie punktom numeru wektora
            if self.method =="cosine":
                m = np.argmax([[wi.T@x for wi in w] for x in X_norm], axis=1)
            elif self.method =="euklidean":
                m = np.argmin([[np.linalg.norm(wi - x) for wi in w] for x in X_norm], axis=1)
            elif self.method =="manhatan":
                m = np.argmin([[sqrt(sum(np.abs(wi - x))) for wi in w] for x in X_norm], axis=1)
            else:
                raise NotImplementedError

            # aktualizacja współczynnika uczenia wg prezentacji (3 opcje)
            if self.alpha_method == "linear":
                alpha = alpha*(iterations - k)/iterations
            elif self.alpha_method == "exponential":
                alpha = alpha_init*exp(-k)
            elif self.alpha_method == "hiperbolic":
                C_1 = 1
                C_2 = 2
                alpha = C_1/(C_2 + k)
            else:
                raise NotImplementedError
            # # aktualizowanie wektorów reprezentantów
            for x, i in zip(X_norm, m):
                w[i] = w[i] + alpha * (x - w[i])
                # normalizacja wektorów rep
                w[i] = w[i] / np.linalg.norm(w[i])
        return m

    # zwracanie tablicy wektorów - każdy wektor zawiera środek ciężkości klastra
    def get_centroids(self, X, iterations, alpha_init):
        m = self.fit_transform(X, iterations, alpha_init)
        # rozdzielenie danych wg przynależności do klastrów
        data_by_cluster = [np.array([sample for sample, cluster_nr in zip(X, m) if cluster_nr == i]) for i in range(self.n_clusters)]
        # obliczanie środków ciężkości klas
        centroids = [cluster.sum(axis=0)/len(cluster) for cluster in data_by_cluster if cluster.any()]
        return centroids


# funkcja aktywacji - zwraca funkcję, z parametrem x, i stałą r
def get_phi(r: float):
    def phi(x: float) -> float:
        return exp(-(x / r) ** 2)
    return phi


# funkcja wyznaczająca średnicę całego zbioru
def diameter(Points):
    diam = max([sqrt(np.sum((p - q) ** 2, axis=None)) for p, q in tqdm.tqdm(combinations(Points, 2))])
    return diam


# aproksymacja na podstawie w, c, r
def get_approx(w, c, r):
    """
    funkcja aproksymująca
    @param w: wektor wag sieci
    @param c: wektor centrów (ma p elementów)
    @param r: średnica całego zbioru
    @return: wyjście sieci y(x)
    """
    phi = get_phi(r)
    return lambda x: np.sum([w[i] * phi(np.linalg.norm(x - c[i])) for i in range(len(c))])


def RBF_iter(X, y, p=20, n_iterations=200, l_r=0.1):
    """
    Algorytm iteracyjny liczenia wag
    @param X: dane wejściowe
    @param y: pożądane wyniki
    @param p: liczba klastrów
    @param n_iterations: ilość powtórzeń
    @param l_r: prędkość uczenia
    @return: wektor wag, wektor centrów, średnica zbioru
    """
    # definicja tablicy F
    F = y

    # klasteryzacja zbioru i obliczenie centroid
    print("Data clustering")
    c = Kohonen(p).get_centroids(X, iterations=100, alpha_init=0.2)
    print("Calculating the set diameter")
    r = diameter(X)

    # konstrukcja funkcji phi
    phi = get_phi(r)

    # początkowe losowe inicjowanie wag
    w = np.random.rand(len(c))

    # aktualizowanie wag - część iteracyjna
    print("Learning")
    for i in tqdm.tqdm(range(n_iterations)):
      approx_fcn = get_approx(w, c, r)
      for sample, label in zip(X, y):
          y_pred = approx_fcn(sample)
          for j in range(len(w)):
              w[j] = w[j] + l_r * (label - y_pred) * phi(np.linalg.norm(sample - c[j]))
    return w, c, r


# WCZYTYWANIE DANYCH
path = "KIM_data.csv"
data = pd.read_csv(path)

X = data[["high", "low", "close", "volume"]].values
y = data[["open"]].values

X_cpy = X.copy()
y_cpy = y.copy()
# 5 dni do jednego
ile_dni = 5
for i in range(len(X)):
    if i - ile_dni >= 0:
        for j in range(1, ile_dni):
            X[i] += X_cpy[i-j]
            y[i] += y_cpy[i - j]
        X[i] = X[i]/ile_dni
        y[i] = y[i] / ile_dni
    else:
        X[i] = "NaN"
        y[i] = "NaN"
# usunięcie niepełnych danych
X = X[5:]
y = y[5:]

# Tworzenie zbioru testowego i treningowego
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# normalizacja
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ------------------------------ wywołanie -------------------------------#

# trening
w, c, r = RBF_iter(X_train, y_train, p=20)
approx_fcn = get_approx(w, c, r)

# # predykcje dla zbioru treningowego
# predictions = [approx_fcn(x) for x in X_train]
# error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_train)]) / len(predictions)
#
# print("MSE - training data = ", error)
# plt.scatter(y_train, predictions)
# plt.show()

# predykcje dla zbioru testowego (czyli ta istotna część)
predictions = [approx_fcn(x) for x in X_test]
error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_test)]) / len(predictions)
error_abs = [(abs(prediction - label)*100)/label for prediction, label in zip(predictions, y_test)]
plt.plot(error_abs)
plt.show()

print("MSE - test data = ", error)
plt.scatter(y_test, predictions)
plt.show()