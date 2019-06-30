# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 11:59:14 2019

@author: Lenovo
"""
import matplotlib.pyplot as plt
import numpy as np

def load_data(path):
    with open(path,"r",encoding="utf-8") as f:
        data = f.readlines()
        data = [i.strip().split("\t") for i in data]
        data = [[i[1].split(" "),i[2].split(" ")]for i in data]
        new_data = []
        for i in data:
            tmp_data = []
            for j in i:
                tmpp_data = []
                for k in j:
                    tmpp_data.append(float(k))
                tmp_data.append(tmpp_data)
            new_data.append(tmp_data)
    return new_data

def plot_precision(data):
    train_p = [i[0][0] for i in data]
    test_p = [i[1][0] for i in data]
    fig=plt.figure()
    plt.yticks(ticks=np.arange(0,1,step=0.2))
    plt.plot(train_p, label="train precision")
    plt.plot(test_p,label="test precision")
    plt.legend(loc='upper left')
    plt.savefig("./precision.png",dpi=200)

def plot_recall(data):
    train_p = [i[0][1] for i in data]
    test_p = [i[1][1] for i in data]
#    fig=plt.figure()
    plt.yticks(ticks=np.arange(0,1,step=0.2))
    plt.plot(train_p, label="train recall")
    plt.plot(test_p,label="test recall")
    plt.legend(loc='upper left')
    plt.savefig("./recall.png",dpi=200)

def plot_f1(data):
    train_p = [i[0][2] for i in data]
    test_p = [i[1][2] for i in data]
#    fig=plt.figure()
    plt.yticks(ticks=np.arange(0,1,step=0.2))
    plt.plot(train_p, label="train f1")
    plt.plot(test_p,label="test f1")
    plt.legend(loc='upper left')
    plt.savefig("./f1.png",dpi=200)    
if __name__ == "__main__":
    path = "./bilstmw2vresult.txt"
    data = load_data(path)
    plot_precision(data)
    plot_recall(data)
    plot_f1(data)