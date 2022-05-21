import tensorflow
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import confusion_matrix
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
import warnings

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
#warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None,
              "max_colwidth", None,
              "display.expand_frame_repr", False)
np.set_printoptions(linewidth=100)

# wczytywanie csv
stonks_df = pd.read_csv('KIM_data.csv', usecols=['open', 'high', 'low', 'close', 'volume'])
# wybranie tylko istotnych kolumn
print("Wczytano bazę.\n")

scaler = MinMaxScaler()
stonks_df[['open', 'high', 'low', 'close', 'volume']] = scaler.fit_transform(stonks_df[['open', 'high', 'low', 'close', 'volume']])
#print(stonks_df)

# tworzenie zbioru treningowego z (100-K)% początkowych próbek i testowego z K% próbek końcowych
howManyRows = len(stonks_df)
K = 10  # ile procent danych to dane
last_training_data = (int) (((100-K)/100)*howManyRows)

stonks_tr = stonks_df[:last_training_data]
print(stonks_tr)
stonks_test = stonks_df[last_training_data:]
stonks_test = stonks_test.reset_index(drop=True)
print(stonks_test)

