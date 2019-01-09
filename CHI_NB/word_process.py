# -*- coding:utf-8 -*-
import collections
import os
import jieba
import numpy as np

np.set_printoptions(threshold=np.nan)
def stop_words():  # 读取停用词
    stop_words = []
    with open('stop_words.txt', encoding='utf-8') as f:
        line = f.readline()
        while line:
            stop_words.append(line[:-1])
            line = f.readline()
    stop_words = set(stop_words)
    return stop_words

def process_file(text100, test_all):
    stopwords_list = stop_words()