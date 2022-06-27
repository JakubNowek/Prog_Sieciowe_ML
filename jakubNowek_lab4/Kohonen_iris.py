import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
from sklearn.manifold import TSNE  # do zmiany wymiarów
from scipy.spatial.distance import cityblock  # do liczenia normy Manhattan


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


def kohonen(p, alpha_0, dane, ile_razy_T=10):

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
    # PRZYGOTOWANIE DANYCH

    # przesunięcie
    for i in range(len(dane)):
        suma += dane.iloc[i]
    offset = suma / N
    # odejmowanie offsetu i normalizacja (dzielenie każdego wektora przez jego normę euklidesową)
    for i in range(len(dane)):
        dane.iloc[i] = (offset-dane.iloc[i])/np.linalg.norm(dane.iloc[i])

    print(dane)
    # inicjalizacja wektorów reprezentantów
    for j in range(p):
        wr_list.append(1/np.sqrt(N)*np.ones(howManyCols))

    # print(wr_list)
    print('Wektory repr przed uczeniem:', wr_list, '\n')

    alpha_k = alpha_0

    # ==================GŁÓWNA PĘTLA ALGORYTMU===================== #
    for k in range(T):
        # wyznaczanie miary i przypisywanie punktom numeru wektora
        for i in range(len(dane)):
            temp = norm1(wr_list, dane, p, i)
            #temp = norm2(wr_list, dane, p, i)
            #temp = norm_manh(wr_list, dane, p, i)
            m_list.append(temp)


        # aktualizowanie wektorów reprezentantów
        #for i in range(len(dane)):
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
    return wr_list


dane_plik = 'iris.data'
iris = pd.read_csv(dane_plik, header=None)
iris = iris.iloc[:, [0, 1, 2]]

# =============================wywołanie======================================= #
p = 3
alpha_0 = 0.1
wektory = kohonen(p, alpha_0, iris)
# ============================================================================= #

# zamiana macierzy array na listę wektorów
for i in range(len(wektory)):
    wektory[i] = wektory[i].tolist()

# NORMALIZACJA IRIS do plotowania na wykresie z wektorami
suma = 0
for i in range(len(iris)):
    suma += iris.iloc[i]
offset = suma / len(iris)
# odejmowanie offsetu i normalizacja (dzielenie każdego wektora przez jego normę euklidesową)
for i in range(len(iris)):
    iris.iloc[i] = (offset - iris.iloc[i]) / np.linalg.norm(iris.iloc[i])

# DAVIES-BOULDIN
# szacowanie rzeczywistej liczby klas (minimum funkcji to najbardziej prawdopodobna liczba klas)
results = {}
search_range = [2, 15]
for i in range(search_range[0], search_range[1]):  # sprawdzamy minimum dla liczby klas do 2 do 15
    kmeans = KMeans(n_clusters=i, random_state=30)
    labels = kmeans.fit_predict(iris)
    db_index = davies_bouldin_score(iris, labels)
    results.update({i: db_index})

# wyznaczanie liczby klas, przy których wskaźnik jest najmniejszy
wskazniki = list(results.values())
# print(wskazniki)
liczba_klas = wskazniki.index(min(wskazniki)) + search_range[0]
print('Oszacowana liczba klas: ', liczba_klas)


# PLOTTOWANIE


fig = plt.figure(figsize=(12, 6))
plt.subplots_adjust(wspace=0.1)
ax = fig.add_subplot(121)
ax.plot(list(results.keys()), list(results.values()))
plt.xlabel("Liczba klastrów")
plt.ylabel("Davies-Bouldin Index")
plt.title("Wartość współczynnika Davies’a-Bouldin’a")
#plt.show()


# plt.subplot(1, 2, 1)
ax = fig.add_subplot(122, projection='3d')
start = [0, 0, 0]

limity = plt.gca()
limity.set_xlim([-1, 2])
limity.set_ylim([-1, 2])
limity.set_zlim([-1, 2])


iris_set = iris.iloc[:50]
iris_ver = iris.iloc[50:100]
iris_vir = iris.iloc[100:150]
for i in range(len(iris_set)):
    ax.scatter(iris_set.iloc[i][0], iris_set.iloc[i][1], iris_set.iloc[i][2], c="red")
for i in range(len(iris_ver)):
    ax.scatter(iris_ver.iloc[i][0], iris_ver.iloc[i][1], iris_ver.iloc[i][2], c='yellow')
for i in range(len(iris_vir)):
    ax.scatter(iris_vir.iloc[i][0], iris_vir.iloc[i][1], iris_vir.iloc[i][2], c='blue')


for i in range(p):
    ax.quiver(start[0], start[1], start[2], wektory[i][0], wektory[i][1], wektory[i][2])
# ax.quiver(start[0], start[1], start[2], wektory[2][0], wektory[2][1], wektory[2][2])
ax.view_init(0,45)
plt.title('Trójwymiarowy widok na klasy i wektory reprezentantów')
tytul = dane_plik + f', liczba klas:{p}, szacowana liczba: {liczba_klas}, alpha: {alpha_0}'
plt.suptitle(tytul)
plt.show()


