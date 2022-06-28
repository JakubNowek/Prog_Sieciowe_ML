from math import sqrt,exp
import tqdm
import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from itertools import combinations


# Algorytm Kohonena robi klasteryzacje. Jest to metoda dokonująca grupowania elementów we względnie jednorodne klasy.
# Innymi słowy algorytm bierze kolejne próbki (X), liczbę klastrów i przypisuje każdej próbce pasujący klaster.
class Kohonenalgorithm:
    # Atrybuty nadawane przy tworzeniu obiektu.
    def __init__(self, p:int, method="cosine", alpha_method = "linear"):
        """ p - liczba skupisk danych """
        self.n_clusters = p
        self.method = method
        self.alpha_method = alpha_method

    # metoda realizująca algorytm Kohonena.
    def fit_transform(self, X, iterations, alpha_init):
        # inicjalizacja współczynnika uczenia z metody __init__().
        alpha = alpha_init
         # obliczanie środka ciężko wg prezentacji.
        s = X.sum(axis=0)/len(X)
        # normalizacja danych wg prezentacji.
        X_normalized = np.array([(s - x)/np.linalg.norm(x, axis=-1) for x in X])
        # inicjalizacja wektorów "w- centroidy" wg prezentacji.
        w = np.random.random((self.n_clusters, len(X[0])))
        w = w /np.linalg.norm(w)

        for k in tqdm.tqdm(range(iterations)):
            # przypisanie numeru klastru do próbki poprzez wyliczenie odległości próbek od skupisk.
            # Najmniejsza odległość od danego klastru spowoduje przypisanie do niego danej próbki.
            if self.method =="cosine":
                m = np.argmax([[wi.T@x for wi in w] for x in X_normalized], axis=1)
            elif self.method =="euklidean":
                m = np.argmin([[np.linalg.norm(wi - x) for wi in w] for x in X_normalized], axis=1)
            elif self.method =="manhatan":
                m = np.argmin([[sqrt(sum(np.abs(wi - x))) for wi in w] for x in X_normalized], axis=1)
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
            # aktualizacja wag wg prezentacji. Wagi (centroidy) są początkowo
            # inicjalizowane w pewnym punkcie, po czym się aktualizują, czyli zbliżają sie do skupisk.
            for x, i in zip(X_normalized, m):
                w[i] = w[i] + alpha*(x - w[i])
                w[i] = w[i] / np.linalg.norm(w[i])

            # m jest tablicą. Kolejne elementy tablicy "m" odpowiadają danym X.
            # w każdym elemencie tablicy m jest numer klastru, do którego została przydzielona dana próbka.
        return m

    # funkcja zwracają tablicę z wyliczonymi środkami ciężkości klastrów.
    def get_centroids(self, X, iterations, alpha_init):
        # wywołanie algorytmu Kohonena. Zdobycie tablicy m, która mówi nam o przynależności próbek do poszczególnych klastrów.
        m = self.fit_transform(X, iterations, alpha_init)
        # podział danych na klastry.
        data_by_cluster = [np.array([sample for sample, cluster_nr in zip(X, m) if cluster_nr == i]) for i in range(self.n_clusters)]
        # obliczanie centroid jako środków cięzkości zbiorów. środek ciężkości jest to środek poszczególnych skupisk.
        centroids = [cluster.sum(axis=0)/len(cluster) for cluster in data_by_cluster if cluster.any()]
        return centroids


# Liczenie funkcji radialnej Phi.
def get_phi(r: float):
    def phi(x: float) -> float:
        return exp(-(x / r) ** 2)

    return phi


# Liczenie objętości zbioru (średnicy - r).
def diameter(Points):
    diam = max([sqrt(np.sum((p - q) ** 2, axis=None)) for p, q in tqdm.tqdm(combinations(Points, 2))])
    return diam


# Aproksymacja na podstawie parametrów parametrów w,c,r zdobytych w funkcji RBF_approximation.
# Aproksymacja jest liczona na podstawie powyższego wzoru.
def get_approximation(w, c, r):
    # konstrukcja funkcji phi (funkcja radialna).
    phi = get_phi(r)
    return lambda x: np.sum([w[i] * phi(np.linalg.norm(x - c[i])) for i in range(len(c))])


""" Algorytm algebraiczny liczenia wag. """
# Funkcja zwraca w - wektor wag, c - centroidy, r - średnica zbioru.
def RBF_approximation(X, y, p=20):
    """ Funkcja oblicza wagi aproksymacji RBF na podstawie danych X, etykiet y i liczby klastów p"""
    # definicja tablicy F
    F = y

    # klasteryzacja zbioru i obliczenie centroid.
    print("Data clustering")
    c = Kohonenalgorithm(p=p).get_centroids(X, iterations=100, alpha_init=0.5)

    # ustalenie parametru r = średnica zbioru.
    r = diameter(X)

    # konstrukcja funkcji phi (funkcja radialna).
    phi = get_phi(r)

    # definicja tablicy Phi wielkiego (ze wzoru w=Phi*F)
    # liczba wierszy = liczba danych x  (1...N)
    # liczba kolumn = liczba centroid   (1...p)
    Phi = np.array([[phi(np.linalg.norm(X[j] - c[i])) for i in range(len(c))] for j in range(len(X))])
    # Obliczenie wektora w na podstawie wzoru w=Phi^(-1)*F. Phi^(-1) - pseudoodwrotność
    w = np.linalg.pinv(Phi) @ F
    return w, c, r


""" Algorytm iteracyjny liczenia wag. """

def RBF_approximation_iter(X, y, p=20, n_iterations=200, l_r=0.1):
  """ Funkcja oblicza wagi aproksymacji RBF na podstawie danych X, etykiet y i liczby klastów p"""
  # definicja tablicy F
  F = y

  # klasteryzacja zbioru i obliczenie centroid
  print("Data clustering")
  c = Kohonenalgorithm(p=p).get_centroids(X, iterations=100, alpha_init=0.2)
  # c = KMeans(n_clusters=p, random_state=0).fit(X).cluster_centers_

  # r = średnica zbioru
  # ustalenie parametru r
  print("Calculating the set diameter")
  r = diameter(X)

  # konstrukcja funkcji phi
  phi = get_phi(r)

  # losowa inicjalizacja wag.
  w = np.random.rand(len(c))

  # aktualizacja wag zgodnie ze wzorem z powyższego obrazka.
  print("Learning")
  for i in tqdm.tqdm(range(n_iterations)):
      aproximation_function = get_approximation(w, c, r)
      for sample, label in zip(X, y):
          y_pred = aproximation_function(sample)
          for j in range(len(w)):
              w[j] = w[j] + l_r * (label - y_pred) * phi(np.linalg.norm(sample - c[j]))

  return w, c, r


path = "KIM_data.csv"
data = pd.read_csv(path)

# Dane wejściowe X
X = data[["high", "low", "close", "volume"]].values
# Dane wyjściowe y (Będziemy opisywać wszystkie ceny otwarcia).
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

X = X[5:]
#X = X.reset_index(drop=True)

y = y[5:]
#y = y.reset_index(drop=True)
# Podział danych na dane testowe oraz dane treningowe.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# normalizacja danych.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# wywołanie algorytmu algebraicznego.
w, c, r = RBF_approximation(X_train, y_train, p=20)
print('WSZMATEN\n', w)
# stworzenie funkcji aproksymującej. Funkcja na podstawie dobranych wag przewidzi wszystkie ceny otwarcia.
aproximation_function = get_approximation(w, c, r)

# predykcja dla danych wejściowych X_train. Predykcja przypiszę cenę otwarcia (klasteryzacja jest działaniem pośrednim w tym wszystkim).
# Predykcja jest to jedynie powielenie wyjść funkcji approximation_function.
predictions = [aproximation_function(x) for x in X_train]
# Błąd średniokwadratowy (mse).
error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_train)]) / len(predictions)

print("Bład mse dla danych treningowych = ", error)

# Wykres ilustrujący poprawność zadziałania algorytmu
# oś x - rzeczywista odpowiedź
# oś y - predykcja
# Gdyby algorytm działał idealnie, wykres byłby idealnie na przekątnej.
plt.scatter(y_train, predictions)
plt.show()

predictions = [aproximation_function(x) for x in X_test]
error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_test)]) / len(predictions)

print("Bład mse dla danych testowych = ", error)
plt.scatter(y_test, predictions)
plt.show()

w, c, r = RBF_approximation_iter(X_train, y_train, p=20)
print('WWWWWWWWWWWWW', w)

aproximation_function = get_approximation(w, c, r)

predictions = [aproximation_function(x) for x in X_train]
error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_train)]) / len(predictions)

print("Bład mse dla danych treningowych = ", error)

plt.scatter(y_train, predictions)
plt.show()

predictions = [aproximation_function(x) for x in X_test]
error = np.sum([(prediction - label) ** 2 for prediction, label in zip(predictions, y_test)]) / len(predictions)

print("Bład mse dla danych testowych = ", error)

plt.scatter(y_test, predictions)
plt.show()