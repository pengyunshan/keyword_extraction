# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 20:47:22 2019

@author: Administrator
"""
import jieba

train_data = "input/train_docs_keywords.txt"
all_data = "input/all_docs.txt"
dic_data = "input/dict.txt"
#生成分词词典
def gene_key_dic():
    with open(train_data,'r',encoding="utf-8") as f:
        lines = f.readlines()
        keys = [line.strip().split("\t")[1].split(",") for line in lines]
    tmp = set()
    for key in keys:
        for i in key:
            tmp.add(i)
    with open(dic_data,'w',encoding="utf-8") as f:
        for i in tmp:
            if (i!=""):
                f.write(i+"\n")
    
# 生成训练数据    
def gene_data():
    with open(train_data,'r',encoding="utf-8") as f:
        lines = f.readlines()
        keys = [line.strip().split("\t")[0] for line in lines]

    with open(all_data,"r",encoding="utf-8") as f,\
        open(train_data,"r",encoding="utf-8") as f2:
        x = []
        line = f.readline()
        while line:
            key = line.strip().split("\x01")[0]
            if key in keys:
                x.append(line.strip().split("\x01"))
            line = f.readline()
        y = []        
        lines = f2.readlines()
        y = [line.strip().split("\t") for line in lines]

        x = sorted(x,key=lambda x:x[0])
        y = sorted(y,key=lambda y:y[0])
    with open("output/x.txt","w",encoding="utf-8") as f1,\
    open("output/y.txt","w",encoding="utf-8") as f2:
        for i in x:
            f1.write(i[1]+"。"+i[2]+"\n")
        for i in y:
            f2.write(i[1]+"\n")
    return x,y

# 生成序列标注数据的x和y
def gene_order_x():
    with open("input/all_docs.txt","r",encoding="utf-8") as f2,\
    open("input/train_docs_keywords.txt","r",encoding="utf-8") as f3,\
    open("output/sentences.txt","r",encoding="utf-8") as f4,\
    open("output/split_x.txt","w",encoding="utf-8") as ft1:
        # 对x进行分词直接调用sentence里的分词结果
        # 先获得all_doc里的文档顺序，因为sentence里的顺序没有存
        all_orders = []
        line = f2.readline()
        while line:
            order = line.strip().split("\x01")[0]
            all_orders.append(order)
            line = f2.readline()
        tmp = f4.readlines()
        sentences = [i.strip().split(" ") for i in tmp]
        sentences_dic = dict()
        for index, order in enumerate(all_orders):
            sentences_dic[order] = sentences[index]
        # 以上获得了一个巨大的dict,把train里有的order排序，然后把对分的分词结果写到split_x
        tmp = f3.readlines()
        keys = [i.strip().split("\t")[0] for i in tmp]
        keys = sorted(keys)
        data = []
        for key in keys:
            senten = sentences_dic[key]
            data.append(senten)
            ft1.write(" ".join(senten)+"\n")
        return data
    
def gene_order_y():     
     # 获得splity的结构  
    with open("output/y.txt","r",encoding="utf-8") as f2,\
    open("output/split_x.txt","r",encoding="utf-8") as ft1,\
    open("output/split_y.txt","w",encoding="utf-8") as ft2:
        # 获得y里的关键词
        keys = []
        lines = f2.readlines()
        for line in lines:
            rows = line.split(",")
            keys.append(rows)
        # 获得x的分词结果
        data = []
        line = ft1.readline()
        while line:
            tmp = line.strip().split(" ")
            data.append(tmp)
            line = ft1.readline()
        # 遍历每一篇新闻
        result = []
        for index,item in enumerate(data):
            # 遍历每一个词
            output = ""
            for word in item:
                if word in keys[index]:
                    output += "K "
                else:
                    output += "o "
            ft2.write(output+"\n")
            result.append(output)
        return result

if __name__ == "__main__":
    gene_key_dic()
    x,y = gene_data()
    #y的关键词最多有6个
    split_x = gene_order_x()
    split_y = gene_order_y()
    