from sklearn import datasets
import pandas as pd

iris = datasets.load_iris()
df_iris_data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
s_iris_target = pd.Series(data=list(map(lambda x: iris.target_names[x], iris.target)), name='target')

df_iris = df_iris_data.join(s_iris_target)
df_iris.to_csv('iris.csv', encoding='utf-8', index=False)
