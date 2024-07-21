from ReadData import ReadData
import torch
import  json
import random
from collections import defaultdict
from torch.utils.data import Dataset,DataLoader
import  torch.nn as nn
import  json
import os
from Model import ESIM
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


def train_model(model,dataloader):

    optimizer=torch.optim.Adam(model.parameters(),lr=0.001)
    epoch=200

    if torch.cuda.is_available():
        model=model.cuda()

    for e in range(epoch):
        for i,data in enumerate(dataloader):
            optimizer.zero_grad()
            if torch.cuda.is_available():
                for i in range(len(data)):
                    data[i]=data[i].cuda()
            premises,premises_lengths,hypotheses,hypotheses_lengths,label=data
            loss=model(premises,premises_lengths,hypotheses,hypotheses_lengths,label)
            loss.backward()
            optimizer.step()
            #打印日志
            print("Epoch:{},step:{},loss:{}".format(e,i,loss.item()))
    torch.save(model,"ESIM.pth")

def eval(readData,model):
    answer2questions=readData.class2question

    if torch.cuda.is_available():
        model=model.cuda()

    readData.mode='eval'
    eval_dataloader=DataLoader(readData,batch_size=1)

    wrong=0
    right=0
    bad_case=[]

    for i,data in enumerate(eval_dataloader):

        if torch.cuda.is_available():
            for i in range(len(data)-1):
                data[i]=data[i].cuda()


        class_probility=defaultdict(int)
        for class_,questions in answer2questions.items():
            premises, premises_lengths, label = data
            questions,lengths=readData.transfer_batch(questions,device='cuda')
            batch=questions.shape[0]
            premises=premises.repeat(batch,1).contiguous()
            premises_lengths=premises_lengths.repeat(batch,1).squeeze().contiguous()
            if premises_lengths.ndim==0:
                class_probility[class_]=0
                continue
            output,similarity=model(premises,premises_lengths,questions,lengths)

            similarity=torch.max(similarity).item()
            class_probility[class_]=similarity

        pred_class=max(class_probility,key=lambda x:class_probility[x])
        if pred_class==data[2][0]:
            right+=1
        else:
            wrong+=1
            bad_case.append(pred_class)
        print("pred_class:{},true_class:{}".format(pred_class,data[2][0]))
    print("acc:{}".format(right/(right+wrong)))



if __name__ == '__main__':
    #train()

    readData = ReadData(epoch_size=1024, negative_probility=0.5)
    readData.LoadData(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json")
    dataloader=DataLoader(readData,batch_size=256,shuffle=True)

    model=ESIM(embedding_size=9526,hidden_size=256,padding_idx=0,dropout=0.5)

    #train_model(model,dataloader)
    model=torch.load("ESIM.pth")
    eval(readData,model)