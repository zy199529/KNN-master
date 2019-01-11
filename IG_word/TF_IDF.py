# -*- coding:utf-8 -*-
from IG_word.IG_reduction import *
from IG_word.word_process import *
from IG_word.SFLA import *
import math

def words():  # 读取降维后的词语
    words = []
    with open('reduction_words.txt', encoding='gb18030', errors='ignore') as f:
        line = f.readline()
        while line:
            words.append(line[:-1])
            line = f.readline()
    return words

def bagOfWord2Vec(vocabList, inputSet):  # 词袋模型，vocabList词典，inputSet每个文本的数据集
    returnVec = []
    word_vec = [0] * len(vocabList)
    Vec = []
    for article in inputSet:  # 对于每一个文本统计词典中的词语出现的个数
        tmp = [0] * len(vocabList)
        vec = [0] * len(vocabList)
        for word in article:  # 对于文本中的每一个单词统计
            if word in vocabList:  # 假如单词出现在词典中
                tmp[vocabList.index(word)] += 1  # 当前文档有这个词条，则根据词典位置获取其位置并赋值为1
                # word_vec[vocabList.index(word)] += 1  # 全部文档的词频，同上，为了方便
                vec[vocabList.index(word)] = 1  # 出现则即为1
        returnVec.append(tmp)  # 统计全部文本的词语频次
        Vec.append(np.array(vec))  # 出现即为1
    return returnVec, Vec


def get_t_idf(df):  # 根据df来计算idf
    N = 1600
    idf_array = []
    for tf in df:
        idf_array.append(math.log10((N + 1) / (tf + 1)) + 1)  # 公式计算idf
    return idf_array


def get_l_tf(tf):
    tf_array = []
    for itf in tf:
        if itf == 0:
            tf_array.append(0.0)
        else:
            tf_array.append(1 + math.log10(itf))  # 根据公式计算
    return tf_array


def tf_idf():
    # 下面计算特征集的TF-IDF的值
    # 使用词袋模型统计词语频次
    vocabList = words()  # 降维后的词典
    label, word_list, class_df_list = fenci_new(file_list="train")  # 分词后1）获得标签2）每个词对应的类别的文本数3）每个文本词典
    returnVec, Vec = bagOfWord2Vec(vocabList, word_list)  # returnVec为词典中词语在每个文本中出现的次数，Vec为只要出现过这个词就为1
    df = np.sum(Vec, axis=0)  # 计算tf_idf中的df词语的频率
    idf_array = get_t_idf(df)  # 计算idf
    idf_array = np.array(idf_array)
    #print("idf的值", idf_array)
    train_vec_List = []
    for sentence in range(np.array(returnVec).shape[0]):
        train_vec_List.append(np.array(get_l_tf(np.array(returnVec)[sentence, :])) * idf_array)  # idf*tf

    return train_vec_List, idf_array, label


def test_tf_idf():
    vocabList = words()  # 降维后的词典
    test_label, test_word_list, class_df_list = fenci_new(file_list="test_A")
    returnVec, Vec = bagOfWord2Vec(vocabList, test_word_list)  # returnVec为词典中词语在每个文本中出现的次数，Vec为只要出现过这个词就为1
    df = np.sum(Vec, axis=0)  # 计算tf_idf中的df词语的频率
    idf_array = get_t_idf(df)  # 计算idf
    idf_array = np.array(idf_array)
    # print("idf的值", idf_array)
    test_vec_List = []
    for sentence in range(np.array(returnVec).shape[0]):
        test_vec_List.append(np.array(get_l_tf(np.array(returnVec)[sentence, :])) * idf_array)  # idf*tf
    return test_vec_List, test_label


if __name__ == '__main__':
    print(test_tf_idf())  # 经过tf_idf后的向量
