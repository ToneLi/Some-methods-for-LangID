# Some-methods-for-LangID
There is a basic test from a Lab：to implement a method to identify the language a document is written in.

## What's LangID?
Language identiﬁcation (LangID) is the task of determining the language(s) that a text is written in. Such as text "杭州西湖是个美丽的地方", its language is Chinese.

## In this work, I decide to use six methods: 
* 1: By using toolkit-langid.
* 2: Bayes+TF-IDF [4]
* 3: Bi-GRU and attention [3]
* 4: Transformer [1]
* 5: Transformer+CNN [5]
* 6: Transformer+CNN+BiGRU
## Enviroment
* Python 3.6.4 
* Tensorflow 1.10.0
* GPU: 2 Telsa P100. 
* Anaconda3-5.1.0-Linux-x86_64.sh
* Ubuntu 16.04.5 LTS

## Corpus
WiLI-2018, the Wikipedia language identification benchmark dataset, contains 235000 paragraphs of 235 languages. The dataset is balanced and a train-test split is provided. In WiLI-2018, the author states: it contains 235000 paragraphs. But it contains duplicated data, duplicated data can influence the result.  In fact, it contains 229095 paragraphs. I randomly divide the data: 90% train, 10% test. Train data: 206185. testdata:22910.[data link](https://zenodo.org/record/841984#collapseCitations). You can gain the processed data by
[BaiDuYu](https://pan.baidu.com/s/13atqh9mWsQIROgH2MIO8IQ) code is "wkdc"

## words or characters
There are 235 languages in WiLI-2018,  the first step about research is to explore the feather about these 235 languages. In NLP, there are several semantic segment styles (word, character, semantic unit which is obtain by searching the knowledge graph, or, lattice). Every language has its own feature, such as Chinese. So this is an challenging task. In this work, I first try to use character feature. Of course, it's also necessary to use the stopword to filter the text.

## the sentence length
After the character segmentation, the max length of all paragraphs is 191974, but, by calculating, 97.31% paragraph is less than 700, so I chose 700 as the max length of paragraph.

## Methods 1
You just need to: pip install langid in your cmd, and then:
```
#encoding=utf-8

import langid

s1 = "中国"
result= langid.classify(s1)
print (result)

#result:('zh', -17.446399450302124)
```
But we should condider do not use an existing library to solve this task.
## Methods 2
This is the standard baseline. In this model, TF-IDF which based on Bags of Words is used for LangID, I used Bayes as a classifier. This baseline was implemented by scikit-learn. My relevent code and result is in the file M2-Bayes. just run the file: detect_language.py.
## Methods 3
This model is build by Bi-GRU and the attention mechanism. This attention mechanism is same as the one which in my paper[2]. My relevent code and result is in the file M3-BiGRU+attention. 
parameter detail:

* word_embedding_dim = 300      #dimension of word embedding
* word_pre_training = None      #use vector_char trained by word2vec
* selfentity_pre_training=None
* word_seq_length = 700         #max length of sentence
* num_classes = 235             #number of labels
* self_entity_seq_length=None
* word_hidden_dim = 300         #the number of hidden units
* word_attention_size =300     #the size of attention layer
* keep_prob = 0.5              #droppout
* learning_rate = 1e-4         #learning rate
* lr_decay = 0.9               #learning rate decay
* clip = 5.0                   #gradient clipping threshold
* num_epochs = 1               #epochs
* batch_size = 8               #batch_size

Notes: I used the randomly initialized embedding matrix by random_uniform, and used the one-hot vector to represent label vecor. You can operate the file "Rnn_Training.py" to run this model. --python Rnn_Training.py or nohup python -u Rnn_Training.py > train.log 2>&1 &
## Method 4-6
In these three models, I mainly foused on the latest model Transformer and model composition. The highlight in this model is muli-head attention and the position information. The authors just used attention to build the model. But I think although they introduced the position embedding in this model, I still think this way cannot obtain the position information in a better way. In github, [5] used Transformer and two layers CNN to enhance the text representation for text classification, I applied his method in LandID. To enhance the position representation, I consider that RNN (Recurrent Neural Network) can also obtain the position information, what would happen if they combined? So I used Bi-GRU to calculate the text's representation, and then combined the information which computed by Transformer+CNN. You can see the detail about this structure in the file "structure.png". On the other hand, you can also learn the propress about the position embedding of transformer in my file "position_embedding.png". In the next, I give a function about how to compute the position embedding.
```
#encoding=utf-8
import tensorflow as tf
sequence_length=5
batch_size=2
embedding_size=10
import numpy as np
position_index = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])
position_embedding = np.array([[pos / np.power(10000, (i - i % 2) / embedding_size) for i in range(embedding_size)] for pos in range(sequence_length)])
"""
__,  __,  __, __, ___  (sequence_length=5), in every position, form a vector, its dimensionality is embedding_size

"""
with tf.Session() as sess:
    sess.run(tf.local_variables_initializer())
    position_index=sess.run(position_index)

    print(position_index)
    print("--------------")
    print(position_embedding)
    print("...>>>>>>>>>>>>>>>>>--")
    print(position_embedding[:, 0::2] )
```


## Result
| Model | Accuracy|
| ------ | ------ |
| Bayes+TF-IDF| 0.8572 |
| Bi-GRU and attention| 0.8069 |
| Transformer| 0.8343|
| Transformer+CNN| 0.8354 |
|Transformer+CNN+BiGRU| 0.8422 |

I am suprised that the transformer have a lower performance than Bayes. May be there are some wrong things when I tuned the model. But, I found that it is feasible to combine Bi-GRU, CNN and transformer.
## Future Work

## Acknowledgement
I would like to thank Prof.Zili Zhou and QFNU for their equipment support.

## Reference
1: Vaswani, Ashish, et al. "Attention is all you need." Advances in neural information processing systems. 2017 [link](http://papers.nips.cc/paper/7181-attention-is-all-you-need)

2：Mingchen Li, Gabtone Clinton, Yi jiaMiao, Feng Gao. Short Text Classiﬁcation via Knowledge powered Attention with Similarity Matrix based CNN, 2019

3: Zhou, Peng, et al. "Attention-based bidirectional long short-term memory networks for relation classification." Proceedings of the 54th annual meeting of the association for computational linguistics (volume 2: Short papers). 2016. [link](https://www.aclweb.org/anthology/P16-2034.pdf)

4: Kononenko, Igor. "Semi-naive Bayesian classifier." European Working Session on Learning. Springer, Berlin, Heidelberg, 1991. [link](https://link.springer.com/chapter/10.1007/BFb0017015)

5: [jiangxinyang227/NLP-Project](https://github.com/jiangxinyang227/NLP-Project/tree/master/text_classifier)
