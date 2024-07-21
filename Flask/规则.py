import jionlp as jio
import jieba
from ReadData import ReadData

readData = ReadData(epoch_size=1024, negative_probility=0.5)
readData.LoadData(r"E:\badouFile\第八周\week8 文本匹配问题\week8 文本匹配问题\data\train.json")

questions=[]
class_=[]

for key,value in readData.class2question.items():
    for sentence in value:
        questions.append(sentence)
        class_.append(key)
questions = [jieba.lcut(text) for text in questions]
print(questions)
print(class_)

result=jio.text_classification.analyse_freq_words(dataset_x=questions,dataset_y=class_,min_word_freq=0.95)


