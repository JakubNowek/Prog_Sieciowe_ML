import pandas as pd
import numpy as np


def kohonen(p, alpha, dane):

    N = len(dane)
    howManyCols = dane.shape[1]
    suma = 0
    wr_list = []

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

    print(wr_list)


iris = pd.read_csv('iris.data', header=None)
iris = iris.iloc[:, [0, 1, 2, 3]]
#iris = iris.to_records(index=False)
#print(iris.iloc[1])


#print(iris)


kohonen(p=3, alpha=0.1, dane=iris)
