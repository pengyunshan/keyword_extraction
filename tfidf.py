# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 10:46:23 2019

@author: i-pengyunshan
"""

#[[,,,],[,,,],[,,,],...]
import math
def gene_dic(data):
    vab = set()
    for i in data:
        for j in i:
            vab.add(j)
    return vab

def get_tf(data):
    result = []
    for i in data:
        vab_file = set(i)
        tmp = dict()
        for w in vab_file:
            num = 0
            for j in i:
                if j == w:
                    num+=1
            tmp[w]=num/len(i)
        result.append(tmp)
    return result

def get_idf(data,vab):
    data = [set(i) for i in data]
    result = dict()
    for i,word in enumerate(vab):
        num = 0
        for file in data:
            if word in file:
                num += 1
        result[word] = num
        if (i%100 ==0):
            print("word:",i)
    return result,len(data)

def get_tfidf(tf,idf,length):
    tf_idf = []
    for i in tf:
        tmp = dict()
        for j in i.keys():
            try:
                tfidf = i[j]*math.log(length/idf[j])
                tmp[j] = tfidf
            except:
                print(j)
        tf_idf.append(tmp)
    return tf_idf

def get_topK(tf_idf, topK):
    result = []
    for i in tf_idf:
        tmp = sorted(i.items(), key=lambda x: x[1], reverse=True)[0:topK]
        tmp = [i[0] for i in tmp]
        result.append(tmp)
    return result
    
if __name__ == "__main__":
#    with open("output/sentences.txt","r",encoding="utf-8") as f:
#        data = f.readlines()
#        data = [i.strip().split(" ") for i in data]
#    with open("output/split_x.txt","r",encoding="utf-8") as f:
#        data_tmp = f.readlines()
#        data_tmp = [i.strip().split(" ") for i in data_tmp]
#    vab = gene_dic(data_tmp)
#    print("length vab:",len(vab))
#    idf,length = get_idf(data,vab)
#    with open("output/idf.txt","w",encoding="utf-8") as f:
#        for i in idf.keys():
#            f.write(i+"\t"+str(idf[i])+"\n")
#    tf = get_tf(data_tmp)
#    with open("output/tf.txt","w",encoding="utf-8") as f:
#        for i in tf:
#            for j in i.keys():
#                f.write(j+"\t"+str(i[j]))
#            f.write("\n")
#    
#    tf_idf = get_tfidf(tf,idf,length)
#    with open("output/tf_idf.txt","w",encoding="utf-8") as f:
#        for i in tf_idf:
#            for j in i.keys():
#                f.write(j+"\t"+str(i[j]))
#            f.write("\n")
    top6=get_topK(tf_idf,6)
    with open("output/result_tf_idf.txt","w",encoding="utf-8") as f:
        for i in top6:
            f.write(",".join(i)+"\n")