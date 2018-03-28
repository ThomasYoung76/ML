"""
    Desion Tree
    entropy：
    f(p) = -log(p)
    H(X) = - (p1 * log(p1) + p2 * log(p2) )
    ID3
"""

import csv
from sklearn.feature_extraction import DictVectorizer
from sklearn import preprocessing
from sklearn import tree
import graphviz

# read data
fa = open('../doc/buy_computers.csv', 'r')
reader = csv.reader(fa)

# features
heads = next(reader)

feature_list = []       # featureas
lable_list = []         # class lales

for row in reader:
    lable_list.append(row[len(row)-1])
    row_dict = {}
    for i in range(1, len(heads)-1):
        row_dict[heads[i]] = row[i]
    feature_list.append(row_dict)

# vectorize features
vector = DictVectorizer()
print(vector)
dummyX = vector.fit_transform(feature_list).toarray()

# vectorize class labels
lb = preprocessing.LabelBinarizer()
dummyY = lb.fit_transform(lable_list)

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(dummyX, dummyY)

# 保存dot文件，使用graphviz转换成pdf
# with open('pt_information_gain.dot', 'w') as f:
#     dot_data = tree.export_graphviz(clf, f, feature_names=vector.get_feature_names())

# 直接使用graphviz库来把文件保存为pdf和dot格式
dot_data = tree.export_graphviz(clf, out_file=None, feature_names=vector.get_feature_names())
graph = graphviz.Source(dot_data)
graph.render('tree')

# 预测
new_rowX = dummyX[0:1]
print(new_rowX)
predictY = clf.predict(new_rowX)
predictY = clf.predict_proba()
print(predictY)
