# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 16:00:17 2019

@author: i-pengyunshan
"""
from datetime import datetime
from sklearn.model_selection  import train_test_split
import torch
import torch.autograd as autograd # torch中自动计算梯度模块
import torch.nn as nn             # 神经网络模块
import torch.nn.functional as F   # 神经网络模块中的常用功能 
import torch.optim as optim       # 模型优化器模块
torch.manual_seed(1)


class LSTMTagger(nn.Module):
 
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()
 
    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))
 
    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, self.hidden = self.lstm(embeds.view(len(sentence), 1, -1), self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores

def get_data():
    with open("output/split_x.txt","r",encoding="utf-8") as f:
        X = f.readlines()
        X = [i.strip().split(" ") for i in X]
    with open("output/split_y.txt","r",encoding="utf-8") as f:
        y = f.readlines()
        y = [i.strip().split(" ") for i in y]
    training_data = []
    for i,_ in enumerate(X):
        training_data.append((X[i],y[i]))
    return training_data

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

def get_to_ix(training_data):
    word_to_ix = {} # 单词的索引字典
    for sent, tags in training_data:
        for word in sent:
            if word not in word_to_ix:
                word_to_ix[word] = len(word_to_ix)
    print(word_to_ix)
    tag_to_ix = {"K": 0, "o": 1} # 手工设定词性标签数据字典
    return word_to_ix,tag_to_ix

def train(epochs,training_data,word_to_ix,tag_to_ix):
    EMBEDDING_DIM =200
    HIDDEN_DIM = 100
    model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
    loss_function = nn.NLLLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    
    for epoch in range(epochs): # 我们要训练300次，可以根据任务量的大小酌情修改次数。
        i = 0
        for sentence, tags in training_data:
            i += 1
            # 清除网络先前的梯度值，梯度值是Pytorch的变量才有的数据，Pytorch张量没有
            model.zero_grad()
            # 重新初始化隐藏层数据，避免受之前运行代码的干扰
            model.hidden = model.init_hidden()
            # 准备网络可以接受的的输入数据和真实标签数据，这是一个监督式学习
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            # 运行我们的模型，直接将模型名作为方法名看待即可
            tag_scores = model(sentence_in)
            # 计算损失，反向传递梯度及更新模型参数
            loss = loss_function(tag_scores, targets)
            loss.backward()
            optimizer.step()
            if (i%100 == 0):
                now = datetime.strftime(datetime.now(),"%m-%d %H:%M:%S")
                print(now,"epoch:",epoch,"num:",i)
        path_name = "./model/lstm"+str(epoch)+".pkl"
        print(path_name)
        torch.save(model, path_name)
        print("model has been saved")         

def test(path,data,word_to_ix):
    model = torch.load(path)
    result = []
    tmp = []
    for i,_ in enumerate(data):
        inputs = prepare_sequence(data[i][0],word_to_ix)
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

def get_y(path):
    with open(path,"r",encoding="utf-8") as f:
        y = f.readlines()
        y = [i.strip().split(" ") for i in y]
    return y

def calculate(y,yp):
    p = []
    r = []
    f1 = []
    for i in range(len(yp)):
        k_num = 0
        p_num = 0
        kp_num = 0
        r_num = 0
        for j in range(len(y[i])):
            if yp[i][j] == "K":
                k_num += 1
                if y[i][j] == "K":
                    p_num += 1
            if y[i][j] == "K":
                kp_num += 1
                if yp[i][j] == "K":
                    r_num += 1
        if (p_num == 0):
            p.append(0)
        else:
            p.append(p_num/k_num)
        if(r_num == 0):
            r.append(0)
        else:
            r.append(r_num/kp_num)
        if ((p_num*kp_num+k_num*r_num) == 0):
            f1.append(0)
        else:
            f1.append(2*p_num*r_num/(p_num*kp_num+k_num*r_num))
    return sum(p)/len(yp),sum(r)/len(yp),sum(f1)/len(yp)
        
if __name__ == "__main__":
    epochs = 200
    # 获得所有数据
    data = get_data()
    word_to_ix,tag_to_ix = get_to_ix(data)
    y = get_y("./output/split_y.txt")
    # 切分训练测试集合
    X_train, X_test, y_train, y_test = train_test_split(data, y, test_size=0.33, random_state=42)
    # 使用训练集训练得到模型并保存
    train(epochs,X_train,word_to_ix,tag_to_ix)
    # 比较不同模型的结果
    result = []
    for i in range(epochs):
        # 训练集的结果
        yp_train = test("./model/lstm"+str(i)+".pkl",X_train,word_to_ix)
        p_t,r_t,f1_t = calculate(y_train,yp_train)
        # 测试集的结果
        yp_test = test("./model/lstm"+str(i)+".pkl",X_test,word_to_ix)
        p,r,f1 = calculate(y_test,yp_test)
        result.append(([p_t,r_t,f1_t],[p,r,f1]))
        now = datetime.strftime(datetime.now(),"%m-%d %H:%M:%S")
        print(now,"epoch:",i)
    
    with open("./output/lstm.txt","r") as f:
        for i,j in result:
            f.write(" ".join(map(str,i)))
            f.write("\t")
            f.write(" ".join(map(str,j)))
            f.write("\n")