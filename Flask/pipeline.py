import jionlp as jio
import jieba
from ReadData import ReadData
import torch
import torch.nn as nn
import json

def MaxReverseMatch(text,bayes_result):

    sentence_length=len(text)
    result=[]
    idx, idy = 0, len(text)
    while sentence_length>0:
        found=False
        match_part=text[idx:idy]

        for classes,key_valueDict in bayes_result.items():
            for key in key_valueDict.keys():
                if match_part==key:
                    result.append([classes,key])
                    idx+=1
                    found=True
                    break
            if found==False:
                idy-=1
                sentence_length-=1


    return result





def eval(Dataset, questions2class):

    # 加载验证集
    valid_data = []
    with open(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\valid.json", encoding='utf-8') as f:
        for line in f.readlines():
            line = line.strip()
            line = json.loads(line)
            valid_data.append(line)



    similarity_func = nn.Softmax(dim=-1)

    # 测试
    # model = torch.load('esim.pth')
    # model = model.cuda()
    correct = 0
    wrong = 0

    wrong_case = []
    count = 0
    for question_pair in valid_data:
        print("question:", count + 1, " ", (count + 1) / len(valid_data))
        result=MaxReverseMatch(question_pair[0],questions2class)
        print(result)



    #     count += 1
    #     max_similarity = 0
    #     max_similarity_answer = None
    #     for train_questions, train_class in questions2class.items():
    #
    #         text1_input = readData.transform_text(question_pair[0])
    #         text1_input = readData.padding_sample(text1_input)
    #
    #         text2_input = readData.transform_text(train_questions)
    #         text2_input = readData.padding_sample(text2_input)
    #
    #         model.eval()
    #         with torch.no_grad():
    #             text1_input = torch.tensor([text1_input]).cuda()
    #             text2_input = torch.tensor([text2_input]).cuda()
    #
    #         # 计算相似度得分
    #         output = model(text1_input, text2_input)
    #         similarity = similarity_func(output)
    #         similarity = similarity[:, -1].item()
    #
    #         if similarity > max_similarity:
    #             max_similarity = similarity
    #             max_similarity_answer = train_class
    #
    #     #  print(max_similarity_answer,answer)
    #     if max_similarity_answer == question_pair[1]:
    #         correct += 1
    #     else:
    #         wrong += 1
    #         wrong_case.append([question_pair[0], question_pair[1], max_similarity, max_similarity_answer])
    # print("ACC:", correct / (wrong + correct))
    #
    # print(wrong_case)



if __name__ == '__main__':

    readData = ReadData(epoch_size=1024, negative_probility=0.5)
    readData.LoadData(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json")

    questions=[]
    class_=[]
    questions2class = {}

    for key,value in readData.class2question.items():
        for sentence in value:
            questions.append(sentence)
            class_.append(key)
            questions2class[sentence] = key
    questions = [jieba.lcut(text) for text in questions]
    print(questions)
    print(class_)

    result=jio.text_classification.analyse_freq_words(dataset_x=questions,dataset_y=class_,min_word_freq=0.95)

    eval(readData,result)



