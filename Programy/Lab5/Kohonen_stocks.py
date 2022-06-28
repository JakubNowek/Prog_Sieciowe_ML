import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE  # do zmiany wymiarów
from scipy.spatial.distance import cityblock  # do liczenia normy Manhattan
import seaborn as sns
from sklearn.model_selection import train_test_split
import tqdm
from itertools import combinations

pd.set_option("display.max_columns", None,
              "max_colwidth", None,
              'display.max_rows', 10,
              "display.expand_frame_repr", False)


def norm1(wr_list, dane, ile_klas, index):
    temp = []
    for m in range(ile_klas):
        # miara pierwsza - iloczyn skalarny
        temp.append(np.dot(wr_list[m], dane.iloc[index]))
    miara = np.amax(temp)
    return temp.index(miara)


def norm2(wr_list, dane, ile_klas, index):
    temp = []
    for m in range(ile_klas):
        # miara druga (norma różnicy)
        temp.append(np.linalg.norm(wr_list[m] - dane.iloc[index]))
    miara = np.amin(temp)
    return temp.index(miara)


def norm_manh(wr_list, dane, ile_klas, index):
    temp = []
    for m in range(ile_klas):
        # miara trzecia - sqrt(manhattan)
        temp.append(math.sqrt(cityblock(wr_list[m], dane.iloc[index])))
    miara = np.amin(temp)
    return temp.index(miara)


def activation(x, r=1):
    return np.exp(- (x / r) ** 2)


def RBFoneshot(dane, centra, ile_klas, F, r):
    N = len(dane)
    PHI = np.empty((N, ile_klas))
    for i in range(N):
        for j in range(ile_klas):
            PHI[i][j] = activation(np.linalg.norm(dane.iloc[i] - centra[j]), r)
    print('PHIIII\n', PHI)
    print('press F', F)
    w = np.linalg.pinv(PHI) @ np.array(F)
    print('WUWUNIO\n', w)
    return w


def predict(test, centra, ile_klas, w, r):
    N = len(test)
    PHI = np.empty((N, ile_klas))
    for i in range(N):
        for j in range(ile_klas):
            PHI[i][j] = activation(np.linalg.norm(test.iloc[i] - centra[j]), r)
    print('PHIIII\n', PHI)
    F = PHI @ w
    return F


def checkPred(pred, real):
    plt.scatter(real, pred)
    plt.show()
    blad = []
    for i in range(len(real)):
        blad.append(abs((real[i]-pred[i])/real[i])*100)
    plt.plot(blad)
    plt.show()



def diameter(Points):
    points = np.array(Points)
    diam = max([np.linalg.norm(p-q) for p,q in tqdm.tqdm(combinations(points, 2))])
    return diam


def kohonen(p, alpha_0, dane, norm, ile_razy_T=10):

    N = len(dane)
    T = ile_razy_T
    howManyCols = dane.shape[1]
    suma = 0
    wr_list = []
    m_list = []
    # współczynniki zmniejszania alpha
    C = 1
    C1 = 1
    C2 = 0.5

    # inicjalizacja wektorów reprezentantów
    for j in range(p):
        wr_list.append((1/math.sqrt(howManyCols)) * np.ones(howManyCols))
        #print("dlugosc wektora = ",  np.linalg.norm(wr_list[j]))

    #print('Wektory repr przed uczeniem:', wr_list, '\n')

    alpha_k = alpha_0

    # ==================GŁÓWNA PĘTLA ALGORYTMU===================== #
    for k in range(T):
        # wyznaczanie miary i przypisywanie punktom numeru wektora
        for i in range(N):
            temp = norm(wr_list, dane, p, i)
            m_list.append(temp)

        # aktualizowanie wektorów reprezentantów
            # aktualizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] + alpha_k * (dane.iloc[i]-wr_list[m_list[i]])
            # normalizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] / np.linalg.norm(wr_list[m_list[i]])

        # #1 zmniejszanie liniowe alpha
        alpha_k = alpha_0*(T-k)/T

        # #2 zmniejszanie wykładnicze alpha
        # alpha_k = alpha_0*math.exp(-C*k)

        # #3 zmniejszanie hiperboliczne alpha
        # alpha_k = C1/(C2 + k)

    #print('Wektory repr po uczeniu:\n', wr_list)

    # zamiana macierzy array na listę wektorów
    for i in range(len(wr_list)):
        wr_list[i] = wr_list[i].tolist()
    return wr_list


# wczytywanie csv
dane_plik = 'KIM_data.csv'
stonks = pd.read_csv(dane_plik, usecols=['open', 'high', 'low', 'close'])
# stonks = pd.read_csv('KIM_data.csv', usecols=['open', 'high', 'low', 'close', 'volume'])

stonks_cpy = stonks.copy()
# 5 dni do jednego
ile_dni = 5
for i in range(len(stonks)):
    if i - ile_dni >= 0:
        for j in range(1, ile_dni):
            stonks.iloc[i] += stonks_cpy.iloc[i-j]
        stonks.iloc[i] = stonks.iloc[i]/ile_dni
    else:
        stonks.iloc[i] = "NaN"

# wyrzucanie nieprzydatnych danych
stonks = stonks.iloc[5:]
stonks = stonks.reset_index(drop=True)

# NORMALIZACJA danych do plotowania na wykresie z wektorami
suma = 0
for i in range(len(stonks)):
    suma += stonks.iloc[i]
offset = suma / len(stonks)
# odejmowanie offsetu
for i in range(len(stonks)):
    stonks.iloc[i] = (offset - stonks.iloc[i]) / np.linalg.norm(offset - stonks.iloc[i])

# PODZIAŁ na zbiór treningowy i testowy

# tworzenie zbioru treningowego z (100-K)% początkowych próbek i testowego z K% próbek końcowych
howManyRows = len(stonks)
K = 10 # ile procent danych chcemy przewidziec
forecast_out = 1#(int)((K/100)*howManyRows)  # ile dni chcemy przewidziec

last_training_data = int(((100-K)/100)*howManyRows)
stonks['Prediction'] = stonks[['open']].shift(-forecast_out)
#print(stonks_df)
#stonks_tr = stonks_df[:last_training_data]

# budowa wejść
X = stonks.drop(columns='Prediction')
X_forecast = X[-forecast_out:]  # ustaw X_forecast na K% dni
X = X[:-forecast_out]  # remove last K% from X
#print('X\n', X)
# budowa wyjść
y = stonks['Prediction']
y = y[:-forecast_out]
#print('y\n', y)
# tworzenie zbiorów treningowych i testowych
X_train = X[:int(len(X)*(100-K)/100)]
X_test = X[int(len(X)*(100-K)/100):]

y_train = y[:int(len(y)*(100-K)/100)]
y_test = y[int(len(y)*(100-K)/100):]

X_test = X_test.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)



# =============================wywołanie======================================= #
p = 20
alpha_0 = 0.5
wektory = kohonen(p, alpha_0, X_train, norm1)

print("WEKTORY\n", wektory)
#print('srednica zbioru', diameter(X))


# wektory_pred = kohonen(p, alpha_0, X_test, norm1)
predicted = predict(X_test, wektory, p, RBFoneshot(X_train, wektory, p, y_train, r=diameter(X)/10), r=diameter(X)/10)
checkPred(predicted, y_test)
error = np.sum([(predicted - label)**2 for prediction, label in zip(predicted, y_test)])/len(predicted)

print("Bład mse dla danych testowych = ", error)
# ============================================================================= #


# # DAVIES-BOULDIN
# # szacowanie rzeczywistej liczby klas (minimum funkcji to najbardziej prawdopodobna liczba klas)
# results = {}
# search_range = [2, 15]
# for i in range(search_range[0], search_range[1]):  # sprawdzamy minimum dla liczby klas do 2 do 15
#     kmeans = KMeans(n_clusters=i, random_state=30)
#     labels = kmeans.fit_predict(X_train)
#     db_index = davies_bouldin_score(X_train, labels)
#     results.update({i: db_index})
#
# # wyznaczanie liczby klas, przy których wskaźnik jest najmniejszy
# wskazniki = list(results.values())
# # print(wskazniki)
# liczba_klas = wskazniki.index(min(wskazniki)) + search_range[0]
# print('Oszacowana liczba klas: ', liczba_klas)
#
#
# # PLOTTOWANIE
#
# fig = plt.figure(figsize=(12, 6))
# plt.subplots_adjust(wspace=0.1)
# ax = fig.add_subplot(121)
# ax.plot(list(results.keys()), list(results.values()))
# plt.xlabel("Liczba klastrów")
# plt.ylabel("Davies-Bouldin Index")
# plt.title("Wartość współczynnika Davies’a-Bouldin’a")
# #plt.show()
#
#
# # plt.subplot(1, 2, 1)
# ax = fig.add_subplot(122, projection='3d')
# start = [0, 0, 0]
#
# limity = plt.gca()
# limity.set_xlim([-1, 1])
# limity.set_ylim([-1, 1])
# limity.set_zlim([-1, 1])
#
#
# for i in range(len(X_train)):
#     ax.scatter(X_train.iloc[i][0], X_train.iloc[i][1], X_train.iloc[i][2])
#
# for i in range(p):
#     ax.quiver(start[0], start[1], start[2], wektory[i][0], wektory[i][1], wektory[i][2])
# ax.view_init(0, 30)
# plt.title('Trójwymiarowy widok na klasy i wektory reprezentantów')
# tytul = dane_plik + f', liczba klas:{p}, szacowana liczba: {liczba_klas}, alpha: {alpha_0}'
# plt.suptitle(tytul)
# plt.show()

