# -*- coding:utf-8 -*-
from IG_word.TF_IDF import *
def words():  # 保存最终降维后的字典
    words = []
    with open('reduction_words.txt', encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            words.append(line[:-1])
            line = f.readline()
    return words

def Jacquard():#计算最大的雅卡尔系数
    vocabList = words()  #降维后的词典
    label, word_list, class_df_list = fenci_new(file_list="train")
    returnVec, Vec = bagOfWord2Vec(vocabList, word_list)#主要是求Vec求每个文本的特征项 0,1
    L = np.zeros((len(label), len(label)))#L矩阵用来存放雅卡尔系数
    for i in range(len(Vec)):
        for j in range(len(Vec)):
            if j > i:  #比较A,B
                A = Vec[i]
                B = Vec[j]
                value = compute_value(A, B)#计算A,B的雅卡尔系数
                L[i][j] = value
    max_value = find_max_value(L)#找到雅卡尔系数最大值得位置然后进行聚类
    Cluster(Vec, max_value)

def Cluster(Vec, max_value):
    for k in range(len(max_value)):
        m, n = max_value[k]
        M = Vec[m]
        N = Vec[n]
        for h in range(len(M)):
            if M[h] == N[h] :



def find_max_value(data_matrix):
    '''
    功能：找到矩阵最大值
    '''
    new_data=[]
    x=[]
    y=[]
    for i in range(len(data_matrix)):
        new_data.append(max(data_matrix[i]))
    max_value = max(new_data)  #找到最大的数
    for i in range(len(data_matrix)):
        for j in range(len(data_matrix.shape[1])):
            if data_matrix[i][j] == max_value:
                x.append((i, j)) #记录最大的位置
    return x

def compute_value(A,B):
    G = np.zeros(len(A))  # G产生一个零向量
    a = A
    b = B
    aa = np.zeros(len(b))  # aa产生一个零向量
    bb = np.zeros(len(b))  # bb产生一个零向量
    k = 0
    for j in range(len(b)):  # 将共同为1的个数保存下来，也将G的相同位置赋值为1，不同的则赋值为2
        if a[j] == b[j]:
            G[j] = 1
            k = k+1  #k即为相同的个数
        else:
            G[j] = 2
    add = np.sum(G)#求出A,B的并集  即G矩阵相加
    Jacquard_value = round(k/add, 2)#雅卡尔系数保留两位有效数字
    return Jacquard_value #返回雅卡尔系数矩阵



