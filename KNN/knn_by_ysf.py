"""
    利用numpy和pandas库实现KNN算法。
"""
__author__ = 'YangShiFu'
__date__ = '2017-11-18'

import numpy as np
import pandas as pd
from doc import utils

def calc_distance(inst1, inst2, length):
    """
    calculate the distance between instance1 and instance2
    :param inst1: array-like
    :param inst2: array-like
    :param length: int 维度
    :return: float
    """
    array_dist = np.power(np.array(inst1[:length]) - np.array(inst2[:length]), 2)
    return np.sqrt(array_dist.sum())

def get_neighbors(training_set, testing_inst, k):
    """
    获取训练集中距离k以内的neighbors
    :param training_set: array-like 训练集
    :param testing_inst: array-like 测试实例
    :param k: int
    :return: 距离k以内的neighbors
    """
    training_set = np.array(training_set)
    testing_inst = np.array(testing_inst)
    # dist = list(map(lambda x: calc_distance(testing_inst, x, length=len(testing_inst)), training_set))
    # training_df = pd.DataFrame(data=training_set)
    # training_df.insert(loc=1, column='dist', value=dist)
    # training_df = training_df.sort_values(by=training_df['dist'], axis=1, ascending=True)
    # training_df = training_df.drop(axis=1, columns='dist')
    # training_df = training_df[:k]  # 取前k行
    # return training_df.as_matrix()
    filt = filter(lambda x: calc_distance(testing_inst, x, length=len(testing_inst)) <= k, training_set) # 过滤出距离小于k
    return np.array(list(filt))

def prediction(neighbors, is_proba=False):
    """
    根据邻近数据预测结果，如果is_proba为false则直接返回结果，否则返回各结果的概率
    :param neighbors: array-like
    :param is_proba: bool whether to calc the probability
    :return:
    """
    df_neighbors = pd.DataFrame(data=neighbors)
    df_count = df_neighbors.groupby(by=df_neighbors.iloc[:,-1]).count()
    s_neighbors = df_count.iloc[:, -1]  # Series
    s_neighbors = s_neighbors.sort_values(ascending=False) # sort descending
    count = []
    for index in s_neighbors.index:
        count.append((index, s_neighbors[index]))
    # predict or predict_probal
    if not is_proba:
        result = [count[0][0], 1]
        return result
    else:
        result = []
        index, values = zip(*count)
        total = sum(values)
        for i in range(len(count)):
            result.append((count[i][0], count[i][1]/total))
        return result


def main():
    training_set =utils.array_from_csv('../doc/iris.csv')
    inst = [3, 4, 4, 1]
    neighbors = get_neighbors(training_set, inst, 3)
    print(prediction(neighbors, is_proba=False))
    print(prediction(neighbors, is_proba=True))

if __name__ == "__main__":
    main()