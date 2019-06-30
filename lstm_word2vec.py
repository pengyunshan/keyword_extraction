# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 20:11:55 2019

@author: Administrator
"""

import lstm
from datetime import datetime
from sklearn.model_selection  import train_test_split
import numpy as np
import collections

import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块
torch.manual_seed(1)

EMBEDDING_DIM =100
HIDDEN_DIM = 100
EPOCHS = 100

def load_my_vocab(path):
    with open(path,"r",encoding = "utf-8") as f:
        data = f.readlines()
        data = [i.strip().split(" ")[0] for i in data]
    return data

# load word embedding
def load_my_vecs(path, vocab):
    word_vecs = collections.OrderedDict()
    with open(path, encoding="utf-8") as f:
        count  = 0
        lines = f.readlines()[1:]
        for line in lines:
            values = line.split(" ")
            word = values[0]
            # word = word.lower()
            count += 1
            if word in vocab:  # whether to judge if in vocab
                vector = []
                for count, val in enumerate(values):
                    if count == 0:
                        continue
                    vector.append(float(val))
                word_vecs[word] = vector
    return word_vecs

# 随机初始化处理未登陆词
def handle_unknow(X,vecs):
    for w in X:
        if w not in vecs.keys():
            vecs[w] = np.random.rand(EMBEDDING_DIM)
    return vecs

class LSTMW2VTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,pretrained_weight):
        super(LSTMW2VTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(pretrained_weight)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = True

        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
 
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def train(training_data,word_to_ix,pretrained_weight):
    tag_to_ix = {"K": 0, "o": 1}    
    model = LSTMW2VTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),pretrained_weight)
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(EPOCHS): # 我们要训练300次，可以根据任务量的大小酌情修改次数。
        i = 0
        for sentence, tags in training_data:
            i += 1
            model.zero_grad()
            model.hidden = model.init_hidden()
            sentence_in = lstm.prepare_sequence(sentence, word_to_ix)
            targets = lstm.prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            if (i%100 == 0):
                now = datetime.strftime(datetime.now(),"%m-%d %H:%M:%S")
                print(now,"epoch:",epoch,"num:",i)
        path_name = "./model/lstm_w2v"+str(epoch)+".pkl"
        print(path_name)
        torch.save(model, path_name)
        print("model has been saved")
        
if __name__ == "__main__":
    # 获得所有数据
    data = lstm.get_data()
    y = lstm.get_y("./output/split_y.txt")
    # 切分训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    vob = set()
    for i in range(len(data)):
        vob = vob.union(set(data[i][0]))
    vecs = load_my_vecs("./output/word2vect.txt",vob)
    vecs = handle_unknow(list(vob),vecs)
    word_to_ix = {}
    tag_to_ix = {"K": 0, "o": 1}
    word_length = len(vecs.keys())
    for i in range(word_length):
        word_to_ix[list(vecs.keys())[i]] = i
    pretrained_weight = list(vecs.values())
    
    train(X_train, word_to_ix,pretrained_weight)
    
    result = []
    for i in range(EPOCHS):
        # 训练集的结果
        yp_train = lstm.test("./model/lstm_w2v"+str(i)+".pkl",X_train,word_to_ix)
        p_t,r_t,f1_t = lstm.calculate(y_train,yp_train)
        # 测试集的结果
        yp_test = lstm.test("./model/lstm_w2v"+str(i)+".pkl",X_test,word_to_ix)
        p,r,f1 = lstm.calculate(y_test,yp_test)
        now = datetime.strftime(datetime.now(),"%m-%d %H:%M:%S")
        print(now,"epoch:",i)
        result.append(([p_t,r_t,f1_t],[p,r,f1]))
    with open("./output/lstmw2vresult.txt","w",encoding="utf-8") as f:
        for i,j in result:
            f.write(" ".join(map(str,i)))
            f.write("\t")
            f.write(" ".join(map(str,j)))
            f.write("\n")