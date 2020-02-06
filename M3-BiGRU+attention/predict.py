#encoding=utf-8
import os
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import batch_iter, sequence
from Rnn_Attention import Rnn_Attention

import  data_helpers
from tensorflow.contrib import learn
import numpy as np
import datetime

def make_source_data(all_file,train_file,test_file):

    print("Loading data...")
    x_all,y_all=data_helpers.load_data_and_labels(all_file)
    x_train,y_train=data_helpers.load_data_and_labels(train_file)
    x_test, y_test = data_helpers.load_data_and_labels(test_file)

    word_level_max_document_length = max([len(x.split(" ")) for x in x_all])  # 最大长度700  经计算 写在下面
    # word_level_max_document_length=34

    vocab_processor = learn.preprocessing.VocabularyProcessor(word_level_max_document_length)
    #
    vocab=vocab_processor.vocabulary_
    #
    # # train  所有的
    x = np.array(list(vocab_processor.fit_transform(x_all)))
    #
    vocab_processor_n = learn.preprocessing.VocabularyProcessor(word_level_max_document_length,vocabulary=vocab)
    # # train
    x_train = np.array(list(vocab_processor_n.fit_transform(x_train)))
    y_train=np.array(y_train)
    # #test
    x_test = np.array(list(vocab_processor_n.fit_transform(x_test)))
    y_test=np.array(y_test)
    return x_train,y_train,vocab_processor,x_test,y_test




def shuffle_data(all_word_file,train_word_file,test_word_file):
    """
    对所有水平的数据进行洗牌 word
    :return:
    """
    x_train_word, y_train_word, vocab_word_processor,x_test_word, y_test_word=make_source_data(all_word_file,train_word_file,test_word_file)

    shuffle_indices = np.random.permutation(np.arange(len(y_train_word)))
    #word
    x_word_shuffled_train = np.array(x_train_word)[shuffle_indices]
    y_word_shuffled_train = np.array(y_train_word)[shuffle_indices]


    return x_word_shuffled_train,y_word_shuffled_train, vocab_word_processor,x_test_word,y_test_word


def val():

    """----word level----"""
    all_word_file = "./data/WiLi2018_space_all.txt"
    train_word_file = "./data/WiLi2018_space_train.txt"
    test_word_file = "./data/WiLi2018_space_test.txt"

    x_word_train, y_word_train, vocab_word_processor, x_word_test, y_word_test = shuffle_data(all_word_file,
                                                                                              train_word_file,
                                                                                              test_word_file)
    pre_label = []
    label = []
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    save_path = tf.train.latest_checkpoint('./checkpoints/Rnn_Attention')
    saver = tf.train.Saver()
    saver.restore(sess=session, save_path=save_path)

    batch_val = batch_iter(x_word_test, y_word_test, batch_size=8)

    for x_batch, y_batch in batch_val:
        seq_length = sequence(x_batch)
        pre_lab = session.run(model.predict, feed_dict = {model.input_word_x1: x_batch,
                                                          model.input_y1: y_batch,
                                                          model.word_seq_length:seq_length,

                                                          model.keep_pro: 1.0})
        pre_label.extend(pre_lab)
        label.extend(y_batch)
    return pre_label, label


if __name__ == '__main__':
    begin=datetime.datetime.now()
    pm = pm
    all_word_file = "./data/WiLi2018_space_all.txt"
    train_word_file = "./data/WiLi2018_space_train.txt"
    test_word_file = "./data/WiLi2018_space_test.txt"

    x_word_train, y_word_train, vocab_word_processor, x_word_test, y_word_test = shuffle_data(all_word_file,
                                                                                              train_word_file,
                                                                                              test_word_file)


    pm.word_vocab_size = len(vocab_word_processor.vocabulary_)

    pm.word_seq_length = x_word_train.shape[1]
    pm.num_classes = y_word_train.shape[1]


    model = Rnn_Attention()

    pre_label, label=val()

    correct = np.equal(pre_label, np.argmax(label, 1))
    accuracy = np.mean(np.cast['float32'](correct))
    print('accuracy:', accuracy)
    print("预测前10项：", ' '.join(str(pre_label[:10])))
    print("正确前10项：", ' '.join(str(np.argmax(label[:10], 1))))
    end=datetime.datetime.now()
    print("the run time is",(end-begin).seconds)


