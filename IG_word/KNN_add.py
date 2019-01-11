# -*- coding:utf-8 -*-
from IG_word.TF_IDF import *

def euclidean(instance1, instance2):  # 计算测试文本和每个训练集文本的距离
    distance = 0
    length = len(instance2)
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def density_compute(trainingSet):  # 训练集的距离矩阵
    dis = np.zeros((len(trainingSet), len(trainingSet)))  # 距离矩阵
    disvec = np.zeros((int((len(trainingSet) * (len(trainingSet) - 1)) / 2), 1))  # 所有距离向量
    k = 1
    for i in range(len(trainingSet)):
        for j in range(i-1):
            dis[i, j] = euclidean(trainingSet[i], trainingSet[j])     # 计算两两文本之间的距离
            dis[j, i] = dis[i, j]
            disvec[k] = dis[i, j]
            k = k+1
    disvec = np.sort(disvec)  # 升序排列
    percent = 0.15
    dc = disvec[round(percent * len(disvec))]  # 选取邻域参数dc,round函数默认保留为整数
    # 计算密度向量p
    p = np.zeros((len(trainingSet), 1))
    for j in range(len(trainingSet)):
        for q in range(len(trainingSet)):
            p[j] += np.exp(-math.pow(dis[j][q]/dc, 2))
        p[j] = p[j] - np.exp(-math.pow(dis[j][j]/dc, 2))  # 高斯核函数 除去自身
    return disvec, p, dc

def test_density_compute(trainingSet, testing, dc):
    dict_value = np.zeros(len(trainingSet))
    for i in range(len(trainingSet)):
            dict_value[i] = euclidean(trainingSet[i], testing)     # 计算两两文本之间的距离
    x = 0
    for j in range(len(trainingSet)):
            x += np.exp(-math.pow(dict_value[j]/dc, 2))
    x = x - np.exp(-math.pow(dict_value[j]/dc, 2))  # 高斯核函数 除去自身
    return x

def test_density():
    train_vec_List, idf_array, label = tf_idf()
    disvec, p, dc = density_compute(train_vec_List)  # 训练集的密度
    # 计算测试文本的密度
    test_vec_List, test_label = test_tf_idf()
    test_density = np.zeros(len(test_vec_List))
    for i in range(len(test_vec_List)):
        test_density[i] = test_density_compute(train_vec_List, test_vec_List[i], dc)
    return test_density

if __name__ == '__main__':
    # 测试文本，使用KNN分类
    print(test_density())