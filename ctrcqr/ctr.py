import jieba
import  numpy as np


class TokenDistance():
    def __init__(self,idf_path):
        idf_dict = {}
        tmp_idx_list=[]
        with open(idf_path,encoding='utf8') as f:
            for line in f:
                ll=line.strip().split(' ')
                idf_dict[ll[0]]=float(ll[1])
                tmp_idx_list.append(ll[0])
        self._idf_dict=idf_dict
        self._media_idf=np.median(tmp_idx_list)
    def predict_jaccard(self,q1,q2):
        if len(q1)<1 or len(q2)<1:
            return 0
        q1=set(list(jieba.cut(q1)))
        q2=set(list(jieba.cut(q2)))
        numerator=sum([self._idf_dict.get(word,self._media_idf) for word in q1.intersection(q2)])
        denominator=sum([self._idf_dict.get(word,self._media_idf) for word in q1.union(q2)])

        return numerator/denominator

    def predict_left(self,q1,q2):
        if len(q1)<1 or len(q2)<1:
            return 0
        q1=set(list(jieba.cut(q1)))
        q2=set(list(jieba.cut(q2)))
        numerator=sum([self._idf_dict.get(word,self._media_idf) for word in q1.intersection(q2)])
        denominator=len(q1)
        return numerator/denominator
    def predict_cqrctr(self,q1,q2):
        if len(q1)<1 or len(q2)<1:
            return 0
        cqr=self.predict_left(q1,q2)
        ctr=self.predict_left(q2,q1)
        return cqr*ctr

if __name__ == '__main__':
    td=TokenDistance('idf.txt')
    print(td.predict_jaccard('我','你'))