import tensorflow
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import dump, load
import warnings
from statistics import median

# ignorowanie warningów (pisało dużo razy, że nie zdążył zbiec)
#warnings.filterwarnings('ignore')
pd.set_option("display.max_columns", None,
              "max_colwidth", None,
              'display.max_rows', None,
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
K = 45  # ile procent danych chcemy przewidziec
forecast_out = 1#(int)((K/100)*howManyRows)  # ile dni chcemy przewidziec

last_training_data = int(((100-K)/100)*howManyRows)
stonks_df['Prediction'] = stonks_df[['open']].shift(-forecast_out)
#print(stonks_df)
#stonks_tr = stonks_df[:last_training_data]

# budowa wejść
X = np.array(stonks_df.drop(columns='Prediction'))
X_forecast = X[-forecast_out:]  # ustaw X_forecast na K% dni
X = X[:-forecast_out]  # remove last K% from X
#print('X\n', X)
# budowa wyjść
y = np.array(stonks_df['Prediction'])
y = y[:-forecast_out]
#print('y\n', y)
# tworzenie zbiorów treningowych i testowych
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=K/100, shuffle=False)


# tworzenie wielowarstwowego perceptronu
mlp = MLPRegressor()

parameters = {
    'hidden_layer_sizes': [20, 40, 60, 80, 100],
    'activation': ['logistic', 'tanh'],
    'learning_rate_init': [0.1, 0.01, 0.001],
    'learning_rate': ['constant', 'adaptive'],
    'solver': ['adam', 'lbfgs']

}

# parameters = {'activation': ['tanh'],
#               'hidden_layer_sizes': [60],
#               'learning_rate': ['constant'],
#               'learning_rate_init': [0.01],
#               'solver': ['adam']}

reg = GridSearchCV(mlp, parameters)

reg.fit(X_train, y_train)
# Testing
confidence = reg.score(X_test, y_test)
print("confidence: ", confidence)

forecast_prediction = reg.predict(X_test)
cvResultsDF = pd.DataFrame(reg.cv_results_)
cvResultsDF = cvResultsDF.sort_values(by=['mean_test_score'])
print("Grid search results: \n", cvResultsDF[["params", "mean_test_score", "rank_test_score"]])
print('Przewidziano na podstawie ', 100-K, '% dni.')

cm = abs(forecast_prediction - y_test)
print('ŚREDNIA BŁĘDÓW', sum(cm)/len(cm))
print('MEDIANA', median(cm))
plt.figure(figsize=(15, 6))
plt.subplots_adjust(wspace=0.3)

plt.subplot(1, 2, 1)
plt.plot(list(range(0, len(cm))), forecast_prediction)
plt.plot(list(range(0, len(cm))), y_test)
plt.title('Przewidziane wartości')
plt.ylabel('Przeskalowana wartość akcji')
plt.xlabel('Numer próbki')
plt.legend(['Predykcja', 'Wartość prawdziwa'])

plt.subplot(1,2,2)
plt.plot(list(range(0, len(cm))), cm)
plt.title('Wartość bezwzględna błędu')
plt.ylabel('Wartość błędu bezwzgl.')
plt.xlabel('Numer próbki')


tytul = "Przewidziano {0} dni (1 dzień do przodu) na podstawie {1} dni."\
        .format(int((K/100)*howManyRows), last_training_data)
plt.suptitle(tytul)
plt.show()







