#encoding=utf-8
import os
import io
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
root_path=os.getcwd()  # this dic
import datetime
# this version is suitable Linux--centos, ubantu, etc...

def loadtrainset(path):

    processed_textset = []
    allclasstags = []
    with io.open(path,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip().split("\t")

            processed_textset.append(line[0])
            allclasstags.append(line[1])
    return processed_textset,allclasstags


if __name__=="__main__":
    begin=datetime.datetime.now()
    train_data, classtags_list=loadtrainset("data_all/WiLi2018_train.txt")
    # print(train_data)
    count_vector = CountVectorizer(lowercase=True,
            analyzer='char_wb',
            ngram_range=(1, 2),
            max_features=1000)
    vecot_matrix = count_vector.fit_transform(train_data)
    # print(train_tfidf)  # vecot_matrix输入，得到词频矩阵
    train_tfidf = TfidfTransformer().fit_transform(vecot_matrix)

    clf = MultinomialNB().fit(train_tfidf, classtags_list)


    true=[]
    predict=[]
    print("----------the training is over-------------")
    with io.open("data_all/WiLi2018_test.txt","r",encoding="utf-8") as fr:
        g=0
        for line in fr.readlines():
            g=g+1
            print(g)
            line=line.strip().split("\t")
            true.append(line[1])
            text=line[0]
            new_count_vector = count_vector.transform([text])
            new_tfidf = TfidfTransformer().fit_transform(new_count_vector)
            train_tfidf = TfidfTransformer().fit_transform(vecot_matrix)
            predict_result = clf.predict(new_tfidf)
            # predict_result = clf.predict(test_tfidf)
            # print(predict_result)
            predict.append(predict_result[0])
    print("------test tuple is over-------")
    j=0
    for i in range(len(predict)):
        if predict[i]==true[i]:
            j=j+1

    print("The accuracy is",j/len(predict))

    end=datetime.datetime.now()
    print("the run time is:",(end-begin).seconds)