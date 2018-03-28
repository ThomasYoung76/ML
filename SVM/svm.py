"""
    suppot vector machine
"""
__author__ = 'YangShiFu'
__date__ = '2017-11-20'

from sklearn.svm import SVC
import numpy as np

# traing set
np.random.seed(0)
X = np.r_[np.random.randn(20, 5) + [2] * 5, np.random.randn(20,5) - [2] * 5]
Y = [0] * 20 + [1] * 20

# build SVM algorithm
clf = SVC(kernel='linear')
clf.fit(X,Y)

# to predict
y = np.random.randn(1,5) - [1] * 5
print(y)
print(clf.predict(y))

print(clf.coef_)
print(clf.support_vectors_)