from IG_word.word_process import *
import operator
from IG_word.IG import *
import numpy


def reduction_words():
    IG_word = feature_selection_ig()  # 返回每个词的信息熵
    floder_path = 'train'  # 分完类的文件夹
    each_word = []  # 存放降维后的词典
    # 该文件夹下宗共有10个文件夹，分别存储10大类的新闻数据
    floder_list = os.listdir(floder_path)  # 读取每一个类别
    for i in range(len(floder_list)):
        word_list = []  # 每个类别的词典
        IG_word_value = []  # 每个类别的词语信息熵
        new_floder_path = floder_path + '/' + floder_list[i]  # 每个类别下面的文件夹名称命名规范
        file_list = eachFile(new_floder_path)  # 读取文件夹名称
        for article in file_list:
            tmp = []
            tmp = create_fenci(article)  # 分词
            word_list.append(tmp)  # 每个类下的文本的词典
        word_list = createVocabList(word_list)  # 每个类别的词典
        for word in word_list:  # 遍历每个类词典的单词
            for k in range(len(IG_word)):  # 遍历之前的信息熵
                if word in IG_word[k][0]:
                    IG_word_value.append(IG_word[k])  # 加入找到该单词，则返回词语和其信息熵
        IG_word_value = list(set(IG_word_value))  # 去重
        IG_word_value.sort(key=operator.itemgetter(1), reverse=True)  # 排序
        for x in range(250):  # 提取前1500个
            each_word.append(IG_word_value[x])  # 提取前1500个单词
    each_word = np.array(list(set(each_word)))
    each_word = each_word[np.lexsort(each_word.T)]
    word = []
    for x in range(len(each_word)):
        word.append(each_word[x][0])
    return word  # 去重词语按信息熵从大到小后返回


if __name__ == '__main__':
    reduction_word = reduction_words()  # 降维后的特征集
    with open('reduction_words.txt', 'w') as f:
        for i in range(len(reduction_word)):
            f.write(str(reduction_word[i]))
            f.write('\n')
    print("结束")