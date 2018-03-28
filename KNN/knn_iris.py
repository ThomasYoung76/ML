"""
    使用sklearn模块实现KNN算法，通过iris中的training datas来预测testing datas
"""
__author__ = 'YangShiFu'
__date__ = '2017-11-18'


import numpy as np
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from doc import utils

# X is training data and y is target values
iris = load_iris()
X,y = iris.data, iris.target

# fit the models using X and y
knn = KNeighborsClassifier(n_neighbors=3, algorithm='brute')
knn.fit(X,y)

# to predict
predict_iris = knn.predict_proba([[3, 4, 4, 1]])
print(predict_iris)
