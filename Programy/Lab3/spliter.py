import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.figure
import matplotlib.pyplot as plt
import random

X = pd.DataFrame({'a': [1,2,3,4,5,6,7,8,9,10], 'b': [1,2,3,4,5,6,7,8,9,10], 'c': [1,2,3,4,5,6,7,8,9,10]})
y = pd.DataFrame({'z': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
print('X_train\n', X_train)
print('X_test\n', X_test)
print('y_train\n', y_train)
print('y_test\n', y_test)

X['Prediction'] = X[['a']].shift(-1)
print('XXXXXXXXX',X)
y = np.array(X['Prediction'])
y = y[:-1]
print(y)
print('SUMA BŁĘDÓW', sum(y))