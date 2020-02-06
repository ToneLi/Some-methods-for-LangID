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
    """
    各种level (word, self-entity, father-entity)  的训练集 测试集
    :return:
    """
    print("Loading data...")
    x_all,y_all=data_helpers.load_data_and_labels(all_file)
    x_train,y_train=data_helpers.load_data_and_labels(train_file)
    x_test, y_test = data_helpers.load_data_and_labels(test_file)
    # print(x_test)
    #
    word_level_max_document_length = max([len(x.split(" ")) for x in x_all])
    # # word_level_max_document_length=700
    # print(word_level_max_document_length)
    vocab_processor = learn.preprocessing.VocabularyProcessor(word_level_max_document_length)
    # #
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
    :return:
    """
    x_train_word, y_train_word, vocab_word_processor,x_test_word, y_test_word=make_source_data(all_word_file,train_word_file,test_word_file)
    # make_source_data(all_word_file, train_word_file, test_word_file)


    shuffle_indices = np.random.permutation(np.arange(len(y_train_word)))
    #char
    x_word_shuffled_train = np.array(x_train_word)[shuffle_indices]
    y_word_shuffled_train = np.array(y_train_word)[shuffle_indices]


    return x_word_shuffled_train,y_word_shuffled_train, vocab_word_processor,x_test_word,y_test_word

def train(x_word_train, y_word_train,
          x_word_test, y_word_test):
    tensorboard_dir = './tensorboard/Rnn_Attention'
    save_dir = './checkpoints/Rnn_Attention'
    if not os.path.exists(tensorboard_dir):  #只是创建目录而已
        os.makedirs(tensorboard_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = os.path.join(save_dir, 'best_validation')


    tf.summary.scalar('loss', model.loss)
    tf.summary.scalar('accuracy', model.accuracy)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(tensorboard_dir)

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)


    saver = tf.train.Saver()
    # session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)



    for epoch in range(pm.num_epochs):
    # # #     print('Epoch:', epoch+1)
        num_batchs = int((len(x_word_train) - 1) / pm.batch_size) + 1
        batch_train = batch_iter(x_word_train, y_word_train,batch_size=pm.batch_size)

        for x_batch,y_batch in batch_train:

            word_seq_len = sequence(x_batch)


            feed_dict = model.feed_data(x_batch, y_batch, word_seq_len, pm.keep_prob)

            _, global_step, _summary, train_loss, train_accuracy = session.run([model.optimizer, model.global_step, merged_summary,
                                                                                model.loss, model.accuracy],feed_dict=feed_dict)

            # print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy)
            if global_step % 100 == 0:
                test_loss, test_accuracy = model.evaluate(session, x_word_test, y_word_test)
                print('global_step:', global_step, 'train_loss:', train_loss, 'train_accuracy:', train_accuracy,
                      'test_loss:', test_loss, 'test_accuracy:', test_accuracy)

            if global_step % num_batchs == 0:
                print('Saving Model...')
                saver.save(session, save_path, global_step=global_step)
    #
    pm.learning_rate *= pm.lr_decay
#     #
#

if __name__ == '__main__':
    # preprocess()
    begin=datetime.datetime.now()
    """----word level----"""
    all_word_file="./data/WiLi2018_space_all.txt"
    train_word_file="./data/WiLi2018_space_train.txt"
    test_word_file="./data/WiLi2018_space_test.txt"

    x_word_train, y_word_train, vocab_word_processor, x_word_test, y_word_test=shuffle_data(all_word_file,train_word_file,test_word_file)
    # shuffle_data(all_word_file, train_word_file, test_word_file)

    pm = pm
    #
    # pm.word_vocab_size =len(vocab_word_processor.vocabulary_)
    pm.word_vocab_size=11185
    # print("word_vocab_size",pm.word_vocab_size) #word_vocab_size 11185
    #
    # pm.word_seq_length=x_word_train.shape[1]
    # pm.num_classes=y_word_train.shape[1]
    #
    print("哥们 开始加载模型了奥！！")

    model = Rnn_Attention()
    print("哥们 模型加载结束 开始进数了。操作起来！")
    train(x_word_train, y_word_train, x_word_test, y_word_test)
    print("卧槽， 累死我了，终于跑完了。喝杯奶。relax!!")
    # print(pm.word_pre_training.shape)


    end = datetime.datetime.now()
    print("CPU为您花费的计算时间为:", (end - begin).seconds, "秒")
