import pandas as pd
import numpy as np

def kohonen(p, alpha_0, dane, ile_razy_T=3):

    N = len(dane)
    T = N * ile_razy_T
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



    for k in range(T):
        # wyznaczanie miary i przypisywanie punktom numeru wektora
        for i in range(len(dane)):
            temp = []
            for m in range(p):
                # miara pierwsza - iloczyn skalarny
                temp.append(np.dot(wr_list[m], dane.iloc[i]))
            miara1 = np.amax(temp)
            m_list.append(temp.index(miara1))

        alpha_k = alpha_0

        # aktualizowanie wektorów reprezentantów
        for i in range(len(dane)):
            # aktualizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] + alpha_k*(dane.iloc[i]-wr_list[m_list[i]])
            # normalizacja
            wr_list[m_list[i]] = wr_list[m_list[i]] / np.linalg.norm(wr_list[m_list[i]])
        # zmniejszanie alpha
        alpha_k = alpha_0*(T-k)/T
    print('Wektory repr po uczeniu:\n', wr_list)


iris = pd.read_csv('iris.data', header=None)
iris = iris.iloc[:, [0, 1, 2]]

kohonen(p=3, alpha_0=0.1, dane=iris)
