# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:36:32 2019

@author: i-pengyunshan
"""

from jieba import analyse
from calculate import get_accuracy
# 引入TextRank关键词抽取接口
def exstranct_keyword():
    textrank = analyse.textrank
    sentence=[]
    with open("output/x.txt","r",encoding="utf-8") as f:
        line = f.readline()
        while line:
            tmp = line.strip()
            sentence.append(tmp)
            line = f.readline()
    keywords6=[]
    for i in sentence:
        keywords = textrank(i,topK=6)
        keywords6.append(keywords)
    with open("output/textrank.txt",'w',encoding="utf-8") as f:
        for i,_ in enumerate(sentence):
            f.write(",".join(keywords6[i])+"\n")
    return keywords6
            
def get_y(path,encoding="utf-8"):
    with open(path,"r",encoding=encoding) as f:
        lines = f.readlines()
        data = [line.strip().split(",") for line in lines]
    return data

if __name__ == "__main__":
#    yp = exstranct_keyword()
#    yp = get_y("output/textrank.txt","gbk")
#    y = get_y("output/y.txt")
#    zhun,zhao,f1= get_accuracy(y,yp)
#    print("zhun:",zhun,"zhao:",zhao,"f1 score:",f1)
#    
    yp = get_y("output/result_tf_idf.txt","utf-8")
    y = get_y("output/y.txt")
    zhun,zhao,f1= get_accuracy(y,yp)
    print("zhun:",zhun,"zhao:",zhao,"f1 score:",f1)