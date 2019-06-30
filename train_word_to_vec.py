# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 21:07:02 2019

@author: Administrator
"""

from gensim.models import Word2Vec
import jieba


def seg_sentence(sentence):  
    jieba.load_userdict("input/dict.txt")
    sentence_seged = jieba.cut(sentence.strip())  
    stopwords = [line.strip() for line in open('input/stopword.txt', 'r', encoding='utf-8')]
    outstr=""
    for word in sentence_seged:  
        if word not in stopwords:  
            if word != '\t':  
                outstr += word  
                outstr += " "  
    return outstr
    
def split_word():
    sentences = []
    with open("input/all_docs.txt",'r',encoding="utf-8") as f,\
       open("output/sentences.txt",'w',encoding="utf-8") as ft:
        lines = f.readlines()
        for i,line in enumerate(lines):
            rows = line.strip().split("\x01")
            tmp = rows[1]+"。"+rows[2]
            tmp = seg_sentence(tmp)
            sentences.append(tmp)
            ft.write(tmp+"\n")
            if (i%100 ==0):
                print(i) 

def get_sentences():
    with open("output/sentences.txt",'r',encoding="utf-8") as f:
        lines = f.readlines()
        sentences = [line.strip().split(" ") for line in lines]
    return sentences

def train_word2vec(sentences):
    model = Word2Vec(sentences, sg=1, size=100,  window=5,  min_count=5,  negative=3, sample=0.001, hs=1, workers=4)
    fname = "model/word2vec.pkl"
    model.save(fname)
    model.wv.save_word2vec_format("output/word2vect.txt","output/vob.txt", binary=False)

    
if __name__ == "__main__":
    sentences = get_sentences()
    train_word2vec(sentences)