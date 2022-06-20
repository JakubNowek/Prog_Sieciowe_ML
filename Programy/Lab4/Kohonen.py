import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score

def kohonen(p, alpha_0, dane, ile_razy_T=10):

    N = len(dane)
    T = ile_razy_T
    howManyCols = dane.shape[1]
    suma = 0
    wr_list = []
    m_list = []
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
    print('Wektory repr przed uczeniem:', wr_list,'\n')

    alpha_k = alpha_0

    # ==================GŁÓWNA PĘTLA ALGORYTMU===================== #
    for k in range(T):
        # wyznaczanie miary i przypisywanie punktom numeru wektora
        for i in range(len(dane)):
            temp = []
            for m in range(p):
                # miara pierwsza - iloczyn skalarny
                temp.append(np.dot(wr_list[m], dane.iloc[i]))
            miara1 = np.amax(temp)
            m_list.append(temp.index(miara1))


        # aktualizowanie wektorów reprezentantów
        #for i in range(len(dane)):
            # aktualizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] + alpha_k*(dane.iloc[i]-wr_list[m_list[i]])
            # normalizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] / np.linalg.norm(wr_list[m_list[i]])

        # zmniejszanie alpha
        alpha_k = alpha_0*(T-k)/T
    #print('Wektory repr po uczeniu:\n', wr_list)
    return wr_list


iris = pd.read_csv('iris.data', header=None)
iris = iris.iloc[:, [0, 1, 2]]

wektory = kohonen(p=3, alpha_0=0.6, dane=iris)

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


# # PLOTTOWANIE dla 3d
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# start = [0, 0, 0]
#
# limity = plt.gca()
# limity.set_xlim([-1, 2])
# limity.set_ylim([-1, 2])
# limity.set_zlim([-1, 2])
#
# for i in range(len(iris)):
#     ax.scatter(iris.iloc[i][0], iris.iloc[i][1], iris.iloc[i][2])
#
# ax.quiver(start[0], start[1], start[2], wektory[0][0], wektory[0][1], wektory[0][2], )
# ax.quiver(start[0], start[1], start[2], wektory[1][0], wektory[1][1], wektory[1][2])
# ax.quiver(start[0], start[1], start[2], wektory[2][0], wektory[2][1], wektory[2][2])
#
# plt.show()

# DAVIES-BOULDIN
# szacowanie rzeczywistej liczby klas (minimum funkcji to najbardziej prawdopodobna liczba klas)
results = {}
search_range = [2, 15]
for i in range(search_range[0], search_range[1]):  # sprawdzamy minimum dla liczby klas do 2 do 15
    kmeans = KMeans(n_clusters=i, random_state=30)
    labels = kmeans.fit_predict(iris)
    db_index = davies_bouldin_score(iris, labels)
    results.update({i: db_index})

plt.plot(list(results.keys()), list(results.values()))
plt.xlabel("Number of clusters")
plt.ylabel("Davies-Bouldin Index")
plt.show()

# wyznaczanie liczby klas, przy których wskaźnik jest najmniejszy
wskazniki = list(results.values())
print(wskazniki)
liczba_klas = wskazniki.index(min(wskazniki)) + search_range[0]
print('Oszacowana liczba klas: ', liczba_klas)
