# -*- coding:utf-8 -*-
from IG_word.KNN_classify import *
from IG_word.tf_idf_sfla import *
import random
from numba import jit,cuda
import numpy as np

def words():  # 读取降维后的词语
    words = []
    with open('reduction_words.txt', encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            words.append(line[:-1])
            line = f.readline()
    return words
@jit
def KNN_classify(second_redution):  # 对每一只鸟巢（词库）KNN分类
    train_vec_List_sfla, idf_array_sfla, label_sfla = tf_idf_sfla(second_redution)  # 计算训练集的tf_idf
    test_vec_List_sfla, test_label_sfla = test_tf_idf_sfla(second_redution)  # 计算测试集的tf_idf
    k = 25
    prediction_sfla = []
    for x in range(len(test_vec_List_sfla)):
        neighbors_sfla = getNeighbors(train_vec_List_sfla, test_vec_List_sfla[x], k, label_sfla)
        result_sfla = getResponse(neighbors_sfla)
        prediction_sfla.append(result_sfla)
        # print("prediction=" + repr(result_sfla) + "  actual=" + repr(test_label_sfla[x]))#预测值和真实值
    accuracy_sfla = getAccurcy(test_label_sfla, prediction_sfla)  # 正确率%
    return accuracy_sfla

@jit
def second_reduction(frog_word):
    frog_vocablist = []
    vocabList = np.array(words())
    for i in range(len(frog_word)):
        if frog_word[i] == 1:
            frog_vocablist.append(vocabList[i])
    accuracy = KNN_classify(frog_vocablist)
    return accuracy
@jit
def CS_init():
    vocabList = words()  # 使用信息熵降维后的词典
    nestNum = 20  # 鸟巢的个数N=20
    T = 10    # 总迭代次数T
    Pa=0.25     # 布谷鸟蛋被发现的概率为 0.25
    myarray = np.random.randint(0, 2, (20, len(vocabList)))    # myarray初始化20个鸟巢的特征项 初始化为 0 或 1
    N, Xg, Xw = compute_fitness(myarray,nestNum)
    Pnest = myarray  # 当前为最优群体
    Bestnest = Xg  # 当前最优鸟巢
    return Pnest, Bestnest, nestNum, vocabList, Pa, T, N
@jit
def compute_fitness(myarray,nestNum):  # 计算每个鸟巢的准确率
    accuracy_all = []
    N = []
    for i in range(nestNum):
        u = second_reduction(myarray[i])
        accuracy_all.append(u)
        N.append(u)
    print(accuracy_all)
    N = np.array(N)
    accuracy_all = np.array(accuracy_all)
    index = np.argsort(accuracy_all)[:: -1]
    return N, index[0], index[19]   # 返回每个鸟巢的准确率，返回最好和最坏的鸟巢
@jit
def CS_best():
    Pnest, Bestnest, nestNum, vocabList, Pa, T ,N = CS_init()
    for t in range(T):
        print("迭代次数", t)
        Pnest_new, N = CS_Iterator(Pnest, Bestnest, nestNum, vocabList, Pa, N)
        Xg = np.argmax(N)
        Bestnest = Xg
        Pnest = Pnest_new
        Xg_array = Pnest_new[Xg]
        print("一次迭代结束,最优的结果", np.max(N))
    return Xg_array
@jit
def CS_Iterator(Pnest, Bestnest, nestNum, vocabList, Pa, N):
    # 返回初始化的当前最优群体，和最好的鸟巢，接下来对鸟巢进行更新
    Pnest_new = Levy(nestNum, vocabList)   # 莱维飞行更新后的鸟巢
    for q in range(nestNum):   # 更新后的鸟巢与上一次的鸟巢进行比较，替换，保留
        old = second_reduction(Pnest[q])    # 对新鸟巢进行遍历
        new = second_reduction(Pnest_new[q])       # 对遍历
        if new >= old:
            Pnest[q] = Pnest_new[q]
            N[q] = new
    print("莱维飞行更新后的新适应度函数", N)
    rand = random.uniform(0, 1)  # 生成随机数，加入大于Pa的话，则替换最坏的那个鸟巢
    print("布谷鸟被发现的概率", rand)
    if rand > Pa:
        index = np.argsort(N)[:: -1]
        Xg_value = index[0]
        Xw_value = index[nestNum-1]
        X_new = Pnest[select_max_fitness(N)]  # 进行轮盘赌选择好的鸟巢
        r1 = np.random.randint(0, 100)  # 生成随机数r1
        r2 = np.random.randint(0, 100)  # 生成随机数r2
        Xw = Pnest[Xw_value]  # 返回最差的那个鸟巢
        Xg = Pnest[Xg_value]  # 返回最好的那个鸟巢
        Xw_new = frog_evolution(X_new, Xw, r1, r2)
        Xw_new_redution = second_reduction(Xw_new)
        Xw_reduction = second_reduction(Xw)
        if Xw_new_redution > Xw_reduction:  # 将产生的新的鸟巢和旧的鸟巢比较适应度函数
            Pnest[Xw_value] = Xw_new  # 若新鸟巢好则用新鸟巢代替
            N[Xw_value] = Xw_new_redution  # 新鸟巢的适应度函数
        else:
             # 否则用全局最优代替局部最优再进行优化
            Xw_new = frog_evolution(Xg, Xw, r1, r2)  # 全局最优进行优化
            Xw_new_redution = second_reduction(Xw_new)
            if Xw_new_redution > Xw_reduction:  # 假如效果好则使用全局最优后选出的青蛙
                Pnest[Xw_value] = Xw_new
                N[Xw_value] = Xw_new_redution
    return Pnest, N
@jit
def frog_evolution(Xb, Xw, r1, r2):
    G = np.zeros(len(Xb))  # G产生一个零向量
    xb = Xb
    xw = Xw
    Xb_Xw = np.zeros(len(xb))  # Xb_Xw产生一个零向量
    Xw_Xb = np.zeros(len(xb))  # Xw_Xb产生一个零向量
    for j in range(len(xb)):  # 将最好和最差的青蛙共同为1的位置保存下来，也将G的相同位置赋值为1，将Xb和Xw
        if xb[j] == xw[j]:
            G[j] = 1
    a = 0
    b = 0
    for k in range(len(xb)):
        if xb[k] == 1 and xw[k] == 0:
            Xb_Xw[k] = 1
            a = a + 1
    a = int(a * r1)  # Xb_Xw需要保留几个1,其余清为0
    for q in range(len(xb)):
        if xw[q] == 1 and xb[q] == 0:
            Xw_Xb[q] = 1
            b = b + 1
    b = int(b * r2)  # Xw_Xb需要保留几个1,其余清为0
    o = 0
    for w in range(len(xb)):  # Xb_Xw保留1,其余清为0
        if Xb_Xw[w] == 1 and o <= a:
            o = o + 1
        else:
            Xb_Xw[w] = 0
    z = 0
    for w in range(len(xb)):  # Xw_Xb保留1,其余清为0
        if Xw_Xb[w] == 1 and z <= b:
            z = z + 1
        else:
            Xw_Xb[w] = 0
    Xw_new = Xw_Xb + Xb_Xw + G  # 新的青蛙产生
    return Xw_new
@jit
def sum(fitness):
    total = 0
    for i in range(len(fitness)):
        total += fitness[i]
    return total
@jit
def select_max_fitness(N): #  使用轮盘赌找出最好的鸟巢
    new_fitness = []
    total_fitness = np.sum(N)  # 适应度的总和
    for i in range(len(N)):  # 概率化,将适应度归一化存入new_fitness
        new_fitness.append(N[i]/total_fitness)
    print("新的适应度：归一化后", new_fitness)
    rand = random.uniform(0, 1)  # 生成随机数
    print("轮盘赌产生的随机数", rand)
    sum = 0
    for g in range(len(N)):  # 轮盘赌

        if rand > sum:
            sum += new_fitness[g]
        else:
            break
    print("轮盘赌选择第几个", g)
    return g
@jit
def Levy(nestNum,vocabList):  # 莱维更新
    a = 1  # 补偿的系数为1
    λ = 2.5  # 参数
    x = 1/λ
    beta = 1.5  # 参数
    sigma_u = math.pow((math.gamma(1 + beta) * math.sin(math.pi * beta / 2)) / ((math.gamma((1 + beta) / 2) * beta * math.pow(2 , (beta - 1) / 2))) , (1 / beta))
    sigma_v = 1
    A = np.zeros((nestNum, len(vocabList)))
    for i in range(nestNum):
        for j in range(len(vocabList)):
            u = random.normalvariate(0, sigma_u)
            v = random.normalvariate(0, sigma_v)
            v = abs(v)
            s = u/math.pow(v, x)
            Step = a * s
            Sig_Step = 1 / (1 + math.exp(-Step))
            rand = random.uniform(0, 1)
            if rand <= Sig_Step:
                A[i][j] = 1
            else:
                A[i][j] = 0  # 更新后当前群体
    return A

if __name__ == '__main__':
    print(CS_best())