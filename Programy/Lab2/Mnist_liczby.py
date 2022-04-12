# https://www.askpython.com/python/examples/load-and-plot-mnist-dataset-in-python

from keras.datasets import mnist
from sklearn.neural_network import MLPClassifier

(train_X, train_y), (test_X, test_y) = mnist.load_data()