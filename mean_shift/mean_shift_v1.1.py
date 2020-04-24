#!/usr/bin/env python3
'''
@File    :   mean_shift.py
@Time    :   2020/04/21 17:33:39
@Author  :   yangshifu
@Version :   v1.1
@Contact :   yangshifu@sensetime.com
@Desc    :   使用meanshift计算视觉中心坐标
'''
import numpy as np
# from matplotlib import pyplot
# from mpl_toolkits.mplot3d import Axes3D
# from sklearn.datasets.samples_generator import make_blobs
from pathlib import Path
import re
from datetime import datetime

"""
方法:
step1: 取一个点当做圆心, 得到所有在半径为bandwidth的圆里的点, 使用这些点计算他们的平均值坐标(对x, y坐标分别求均值), 即质心.
step2: 以质心当圆心, 得到所有在半径为bandwidth的圆里的点, 使用这些点计算他们的质心坐标. 
step3: 重复step2, 直到下一个质心坐标与圆心坐标完全一样. 
step4: 遍历所有的定位点, 重复step1到step3. 
"""

class Mean_Shift():
    
    def __init__(self, bandwidth=0.5):
        self._bandwidth = bandwidth

    def fit(self, data):
        scores = {}
        centers = {}
        for i in range(len(data)):
            center = data[i]
            while True:
                # 把所有圆内的点放入_in_bandwidth
                _in_bandwidth = []
                for feature in data:
                    distance = np.linalg.norm(center-feature)
                    if int(distance // self._bandwidth) == 0:
                        _in_bandwidth.append(feature)
                        scores[i] = len(_in_bandwidth)

                # 找到质心坐标
                new_center = np.average(_in_bandwidth, axis=0)  
                # 质心和圆心为同一点时
                if not (new_center - center).all():
                    centers[i] = new_center 
                    break
                center = new_center
        index = sorted(scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        self.center = centers[index]

                
def get_data(file_name):
    result_xz = np.genfromtxt(file_name, dtype=str, usecols=[0, 1,2], delimiter=',')
    for i in range(len(result_xz)):
        result_xz[i][0] = Path(result_xz[i][0]).parent.name.split('_')[-3:][0]
    # ms.fit()
    dict_xz_point = {}
    dict_xz_area = {}
    for row in result_xz:
        # dict_xz_point的key是点
        if row[0] not in dict_xz_point.keys():
            dict_xz_point[row[0]] = [list(map(float, row[1:]))]
        else:
            dict_xz_point[row[0]].append(list(map(float, row[1:])))
        # dict_xz_area的key是区
        area_id = re.match(r'[A-Za-z]+', row[0]).group()
        if area_id not in dict_xz_area.keys():
            dict_xz_area[area_id] = [list(map(float, row[1:]))]
        else:
            dict_xz_area[area_id].append(list(map(float, row[1:])))
    # for point in dict_xz_point:
    #     print(point)
    #     data = dict_xz_point[point]
    #     break
    return dict_xz_point

if __name__ == "__main__":
    ms = Mean_Shift(0.5)
    file_name = r"/home/SENSETIME/yangshifu/Documents/gitlab/armap/result/ret_xzy_20200421.txt"
    dict_xz_point = get_data(file_name)
    fa = open("GT_center.csv", 'w')
    for point in dict_xz_point:
        data = np.array(dict_xz_point[point])
        start = datetime.now()
        ms.fit(data)
        end = datetime.now()
        print(f"{point},{ms.center[0]},{ms.center[1]},0", file=fa)
        # print(f"循环次数: {ms.count}")
        elapsed = (end - start).seconds
        print(f"points count: {len(data)}. elpased: {elapsed}s")
