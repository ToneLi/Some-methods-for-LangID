#encoding=utf-8

def reload_lable_vecor():
    labels=[]
    vectors=[]
    with open("./data/label_vec.txt","r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip().split("\t")
            label=line[0]
            labels.append(label)
            vector=[float(x) for x in line[1].split(" ")]
            vectors.append(vector)
    dic_=dict(zip(labels,vectors))
    return dic_


def load_data_and_labels(file):
    dic_=reload_lable_vecor()
    sentences=[]
    label=[]
    with open(file,"r",encoding="utf-8") as fr:
        for line in fr.readlines():
            line=line.strip().split("\t")
            # print(line)
            sentence=line[0]
            se_sentence=sentence.split(" ")
            if len(se_sentence)<=700:
                sentences.append(sentence)
            else:
                sentences.append(sentence[0:700])

            label.append(dic_[line[1]])

    return sentences,label