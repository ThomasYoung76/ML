"""
    定义通用方法
"""
__author__ = 'YangShiFu'
__date__ = '2017-11-18'

import numpy as np
import pandas as pd

def array_from_csv(filename, encoding='utf-8'):
    """
        load data from csv str_file. and return a tuple
    :param filename: csv str_file
    :param encoding:
    :return:  ndarray
    """
    df = pd.read_csv(filename,  encoding=encoding)
    return df.as_matrix()


if __name__ == "__main__":
    # test X， y
    data = array_from_csv('buy_computers.csv')
    X,y = data[:, :-1], data[:, -1]
    print(X)
    print(y)
