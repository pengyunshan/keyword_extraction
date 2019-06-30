# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 15:01:50 2019

@author: i-pengyunshan
"""
def calcualte_two_list(y,yp):
    # y=["w","w"]
    # yp=["w","w","w"]
    # 默认y的个数小于等于yp
    len_y = len(y)
    len_yp = len(yp)
    # 准确率，遍历yp里前len_y个：
    zhun = 0
    num = 0
    for i in range(len_y):
        if yp[i] in y:
            num += 1
    zhun += num/len_y
    # 召回率，遍历y里前len_yp个：
    zhao = 0
    num = 0
    for i in range(min(len_y,len_yp)):
        if y[i] in yp[0:len_y]:
            num += 1
    zhao += num/len_yp        
    return zhun,zhao
    
def get_accuracy(y,yp):
    zhun_all = 0
    zhao_all = 0
    for i in range(len(y)):
        zhun,zhao = calcualte_two_list(y[i],yp[i])
        zhun_all += zhun
        zhao_all += zhao
    return zhun_all/len(y),zhao_all/len(y),2*(zhun_all/len(y))*(zhao_all/len(y))/((zhun_all/len(y))+(zhao_all/len(y)))

if __name__ == "__main__":
    with open("output/y.txt","r",encoding="utf-8") as f:
        y = f.readlines()
        y = [i.strip().split(",") for i in y]
    with open("output/result_tf_idf.txt","r",encoding="utf-8") as f:
        yp = f.readlines()
        yp = [i.strip().split(",") for i in yp]
    
    zhun,zhao,f1 = get_accuracy(y,yp)
    print(zhun,zhao,f1)
    with open("output/textrank.txt","r") as f:
        yp = f.readlines()
        yp = [i.strip().split(",") for i in yp]
    zhun,zhao,f1 = get_accuracy(y,yp)
    print(zhun,zhao,f1)
