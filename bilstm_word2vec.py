# -*- coding: utf-8 -*-
"""
Created on Sun Apr 14 14:04:10 2019

@author: Administrator
"""
import lstm
import lstm_word2vec

import sys
from datetime import datetime
from sklearn.model_selection  import train_test_split
import numpy as np

import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块
torch.manual_seed(1)

EMBEDDING_DIM =100
HIDDEN_DIM = 40
EPOCHS = 100

class BILSTMW2VTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size,pretrained_weight):
        super(BILSTMW2VTagger, self).__init__()
        self.hidden_dim = hidden_dim
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        pretrained_weight = np.array(pretrained_weight)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        self.hidden2tag = nn.Linear(hidden_dim * 2, tagset_size)
        self.hidden = self.init_hidden()
        print(self.hidden[0].size())
        
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)))
 
    def forward(self, sentence):
        embeds = self.embedding(sentence)
        
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores
    
def test(model,data,word_to_ix):
    result = []
    tmp = []
    for i,_ in enumerate(data):
        inputs = lstm.prepare_sequence(data[i][0],word_to_ix)
        tag_scores = model(inputs)
        tmp.append(tag_scores)
    for item in tmp:
        tmpp = []
        for i,j in item:
            if (i>j):
                tmpp.append("K")
            else:
                tmpp.append("o")
        result.append(tmpp)
    return result

def test_weight(model,data,word_to_ix):
    result = []
    tmp = []
    for i,_ in enumerate(data):
        inputs = lstm.prepare_sequence(data[i][0],word_to_ix)
        tag_scores = model(inputs)
        tmp.append(tag_scores)
        result.append(data[i][0])
    return tmp

def train(model,training_data,word_to_ix,pretrained_weight,start_epoch=0):
    print("load data finished,begin to train")
    tag_to_ix = {"K": 0, "o": 1}    
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    if start_epoch == 0:
        write_path = "w"
    else:
        write_path = "a"
        
    f = open("./output/bilstmw2vresult.txt",write_path,encoding="utf-8")
    for epoch in range(start_epoch,EPOCHS):
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
        path_name = "./model/bilstm_w2v"+str(epoch)+".pkl"
        torch.save(model, path_name)
        
        yp_train = test(model,X_train,word_to_ix)
        yp_test = test(model,X_test,word_to_ix)
        p_t,r_t,f1_t = lstm.calculate(y_train,yp_train)
        p,r,f1 = lstm.calculate(y_test,yp_test)
        train_result = [p_t,r_t,f1_t]
        test_result = [p,r,f1]
        f.write(str(epoch)+"\t")
        f.write(" ".join(map(str,train_result)))
        f.write("\t")
        f.write(" ".join(map(str,test_result)))
        f.write("\n")
        f.flush()
        print("model has been saved")
        sys.stdout.flush()
    f.close()


if __name__ == "__main__":
    # 获得所有数据
    data = lstm.get_data()
    y = lstm.get_y("./output/split_y.txt")
    # 切分训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    vob = set()
    for i in range(len(data)):
        vob = vob.union(set(data[i][0]))
    vecs = lstm_word2vec.load_my_vecs("./output/word2vect.txt",vob)
    vecs = lstm_word2vec.handle_unknow(list(vob),vecs)
    word_to_ix = {}
    tag_to_ix = {"K": 0, "o": 1}
    word_length = len(vecs.keys())
    for i in range(word_length):
        word_to_ix[list(vecs.keys())[i]] = i
    pretrained_weight = list(vecs.values())
    # 生成模型并训练
    path = "./model/bilstm_w2v99.pkl"
    model = torch.load(path)
#    model = BILSTMW2VTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix),pretrained_weight)
#    train(model,X_train, word_to_ix,pretrained_weight,start_epoch=47)

#    radio,words = test_weight(model,X_test,word_to_ix)