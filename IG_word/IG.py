# -*- coding:utf-8 -*-
from IG_word.word_process import *


def createVocabList(dataSet):  # 统计词汇，创建词典
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)  # 根据数据集返回词典


def vocablist():  # 返回计算信息熵的参数：
    # 计算信息熵需要知道三个数组
    # 1）class_df_list每个类别的文档数
    # 2）term_set词典
    # 3)term_class_df_mat每个单词的在对应类别中出现的文档数
    # 链接：https://zhuanlan.zhihu.com/p/23199165
    label, word_list, class_df_list = fenci_new(file_list="train")
    term_set = createVocabList(word_list)  # 词典
    print(len(term_set))
    term_class_df_mat = []
    for i in range(len(term_set)):
        df_mat = np.zeros(8)
        for j in range(len(label)):
            if label[j] == 1:
                if term_set[i] in word_list[j]:
                    df_mat[0] += 1
            if label[j] == 2:
                if term_set[i] in word_list[j]:
                    df_mat[1] += 1
            if label[j] == 3:
                if term_set[i] in word_list[j]:
                    df_mat[2] += 1
            if label[j] == 4:
                if term_set[i] in word_list[j]:
                    df_mat[3] += 1
            if label[j] == 5:
                if term_set[i] in word_list[j]:
                    df_mat[4] += 1
            if label[j] == 6:
                if term_set[i] in word_list[j]:
                    df_mat[5] += 1
            if label[j] == 7:
                if term_set[i] in word_list[j]:
                    df_mat[6] += 1
            if label[j] == 8:
                if term_set[i] in word_list[j]:
                    df_mat[7] += 1
        term_class_df_mat.append(df_mat)
    return class_df_list, term_set, term_class_df_mat


def feature_selection_ig():  # 计算每个词的信息熵，按信息熵从大到小返回词语
    class_df_list, term_set, term_class_df_mat = vocablist()
    term_class_df_mat = np.array(term_class_df_mat)
    A = term_class_df_mat
    B = np.array([(sum(x) - x).tolist() for x in A])  # 与A对称
    C = np.tile(class_df_list, (A.shape[0], 1)) - A  # 未出现在分类中的词集多维数组
    N = sum(class_df_list)
    D = N - A - B - C  # 与C对称
    term_df_array = np.sum(A, axis=1)
    class_set_size = len(class_df_list)

    p_t = term_df_array / N  # 每个词存在的情况下的概率矩阵
    p_not_t = 1 - p_t  # 每个词缺失的情况下的概率矩阵
    # 避免有些词在某些分类中没有出现过而概率为0的情况，采用拉普拉斯平滑定理：分子加1分母加2平滑法。
    p_c_t_mat = (A + 1) / (A + B + class_set_size)
    p_c_not_t_mat = (C + 1) / (C + D + class_set_size)
    p_c_t = np.sum(p_c_t_mat * np.log(p_c_t_mat), axis=1)
    p_c_not_t = np.sum(p_c_not_t_mat * np.log(p_c_not_t_mat), axis=1)
    IG_word = []
    term_score_array = p_t * p_c_t + p_not_t * p_c_not_t
    # 按term_score_array升序排列，然后倒序，因为计算公式中忽略掉-号，其实也就是得出降序后的索引位置
    sorted_term_score_index = term_score_array.argsort()[:: -1]
    for index in sorted_term_score_index:
        IG_word.append((term_set[index], term_score_array[index]))
    return IG_word


if __name__ == '__main__':
    IG_word = feature_selection_ig()  # 返回每个词的信息熵
    print(IG_word)
