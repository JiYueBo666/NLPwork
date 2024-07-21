from torch.utils.data import Dataset,DataLoader
import torch
import  json
import random
from collections import defaultdict


class ReadData(Dataset):
    def __init__(self,epoch_size,negative_probility=0.5):
        self.class2question={}
        self.epoch_size=epoch_size
        self.mode='train'
        self.negative_probility=negative_probility
        self.sample_length=0
        self.vocab=self.get_vocab(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\chars.txt")
        self.eval_path=r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\valid.json"
        self.eval_question=[]
        self.eval_answer=[]
        self.load_eval_dataset()

    def get_vocab(self,path):
        word_dict=defaultdict(int)
        with open(path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=line.strip()
                word_dict[line]=len(word_dict)
        return word_dict

    def transform_text(self,text):
        text=[self.vocab[char] for char in text]
        return text

    def LoadData(self,path,mode='train'):
        with open(path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=json.loads(line)
                self.class2question[line['target']]=line['questions']
                if mode=='train':
                    #计算样本长度
                    for question in line['questions']:
                        self.sample_length=max(self.sample_length,len(question))
            f.close()

    def padding_sample(self,text):
        if self.sample_length>0:
            if len(text)>self.sample_length:
                text=text[:self.sample_length]
            elif len(text)<self.sample_length:
                text=text+[0]*(self.sample_length-len(text))
            return text

    def transform_label(self):
        self.Schem={}
        count=0
        if len(self.class2question.keys())!=0:
            for key in self.class2question.keys():
                self.Schem[key]=count
                count+=1
        else:
            #抛出异常，加载数据
            raise Exception("数据加载失败，没有初始化数据集")
    def __len__(self):
        if self.mode=='train':
            return self.epoch_size
        if self.mode=='eval':
            return len(self.eval_question)

    def __getitem__(self, item):
        if self.mode=='train':
            text1,text2,label,length1,length2=self.random_sample()
            return torch.tensor(text1),torch.tensor(length1),torch.tensor(text2),torch.tensor(length2),torch.tensor([label])

        elif self.mode=='eval':
            text,answer,text_length=self.sample_eval(item)
            return torch.tensor(text),torch.tensor(text_length),answer


    def load_eval_dataset(self):
        with open(self.eval_path,'r',encoding='utf-8') as f:
            for line in f.readlines():
                line=json.loads(line)
                self.eval_answer.append(line[1])
                self.eval_question.append(line[0])

    def sample_eval(self,idx):
        text=self.eval_question[idx]
        text= self.transform_text(text)
        true_length1 = len(text)
        text=self.padding_sample(text)
        return text,self.eval_answer[idx],true_length1

    def transfer_batch(self,texts,device='cuda'):
        transfer_texts=[]
        texts_length=[]
        for question in texts:
            question=self.transform_text(question)
            texts_length.append(len(question))
            question=self.padding_sample(question)
            transfer_texts.append(question)
        return torch.tensor(transfer_texts).to(device),torch.tensor(texts_length).to(device)

    def random_sample(self):
        label_list=list(self.class2question.keys())
        true_length1=0
        true_length2=0

        #选择正类
        if random.random()<self.negative_probility:
            # 随机选取一个类
            current_label = random.choice(label_list)
            #从该类中选取两个句子作为正样本.

            if len(self.class2question[current_label])<2:
                return self.random_sample()
            else:
                text1,text2=random.sample(self.class2question[current_label],2)
                text1,text2=self.transform_text(text1),self.transform_text(text2)
                true_length1=len(text1)
                true_length2=len(text2)
                text1,text2=self.padding_sample(text1),self.padding_sample(text2)
                return text1,text2,1,true_length1,true_length2
        else:
            #随机选取两个不同类
            class1,class2=random.sample(label_list,2)
            text1,text2=random.choice(self.class2question[class1]),random.choice(self.class2question[class2])
            text1, text2 = self.transform_text(text1), self.transform_text(text2)
            true_length1 = len(text1)
            true_length2 = len(text2)
            text1, text2 = self.padding_sample(text1), self.padding_sample(text2)
            return text1,text2,0,true_length1,true_length2




if __name__ == '__main__':
    readData=ReadData(epoch_size=100,negative_probility=0.5)
    readData.LoadData(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json")
    readData.mode='eval'
    dataLoader=DataLoader(readData,batch_size=1)

    for i,data in enumerate(dataLoader):

        question,length,answer=data
        print(question.shape)
        print(length.shape)

