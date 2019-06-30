# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 19:13:02 2019

@author: i-pengyunshan
"""

import preprocess
import train_word_to_vec
import tfidf
# 按照label生成key分词用
#preprocess.gene_key_dic()
## 提取有用的x和y，并去掉无关的文章顺序，供分类算法用
#x,y = preprocess.gene_data()
## 对所有文本进行分词
#train_word_to_vec.split_word()
# 训练词向量
#sentences = train_word_to_vec.get_sentences()
#train_word_to_vec.train_word2vec(sentences)
## 生成分词的x和y，供序列标注算法使用
#split_x = preprocess.gene_order_x()
#split_y = preprocess.gene_order_y()

vab = tfidf.gene_dic(sentences)
idf, length = tfidf.get_idf(sentences, vab)