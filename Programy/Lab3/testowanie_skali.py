from sklearn import preprocessing
import numpy as np

x = np.array([0.55, 0.01, 0.05, 0.05, 0.07])

x.reshape(-1, 1)

minmax_scale = preprocessing.MinMaxScaler(feature_range=(21, 25))

x_scale = minmax_scale.fit_transform(x)

print(x)
print(x_scale)