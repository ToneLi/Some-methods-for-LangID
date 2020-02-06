#encoding=utf-8
import os
import pickle
import numpy as np

class EvalData():
    def __init__(self, config):
        # super(EvalData, self).__init__(config)

        self._eval_data_path = config["eval_data"]
        self._output_path =  config["output_path"]
        self._dic_path = config["char_dic"]
        self._label_list_path = config["label_list"]

        self._sequence_length = config["sequence_length"]  # 每条输入的序列处理为定长

    def read_data(self):
        """
        读取数据
        :return: 返回分词后的文本内容和标签，inputs = [[]], labels = []
        """
        inputs = []
        labels = []
        with open(self._eval_data_path, "r", encoding="utf8") as fr:
            for line in fr.readlines():
                try:
                    text, label = line.strip().split("\t")
                    inputs.append(text.strip().split(" "))
                    labels.append(label)
                except:
                    continue

        return inputs, labels


    def gen_word_label_index(self):
        """
        生成词汇，标签等映射表
        :param words: 训练集所含有的单词
        :param labels: 标签
        :return:
        """
        vocab = []
        with open(self._dic_path, "r", encoding="utf-8") as fr:
            for line in fr.readlines():
                try:
                    line = line.strip().split("\t")
                    vocab.append(line[0])

                except:
                    continue

        vocab = ["<PAD>", "<UNK>"] + vocab
        self.vocab_size = len(vocab)
        word_to_index = dict(zip(vocab, list(range(len(vocab)))))

        labels = []
        with open(self._label_list_path, "r", encoding="utf-8") as fr1:
            for line1 in fr1.readlines():
                try:
                    line1 = line1.strip()
                    labels.append(line1)

                except:
                    continue

        label_to_index = dict(zip(labels, list(range(len(labels)))))

        return word_to_index, label_to_index

    @staticmethod
    def trans_to_index(inputs, word_to_index):
        """
        将输入转化为索引表示
        :param inputs: 输入
        :param word_to_index: 词汇-索引映射表
        :return:
        """
        inputs_idx = [[word_to_index.get(word, word_to_index["<UNK>"]) for word in sentence] for sentence in inputs]

        return inputs_idx

    @staticmethod  # do not need self anymore
    def trans_label_to_index(labels, label_to_index):
        """
        将标签也转换成数字表示
        :param labels: 标签
        :param label_to_index: 标签-索引映射表
        :return:
        """
        labels_idx = [label_to_index[label] for label in labels]
        return labels_idx

    def padding(self, inputs, sequence_length):
        """
        对序列进行截断和补全
        :param inputs: 输入
        :param sequence_length: 预定义的序列长度
        :return:
        """
        new_inputs = [sentence[:sequence_length]
                      if len(sentence) > sequence_length
                      else sentence + [0] * (sequence_length - len(sentence))
                      for sentence in inputs]

        return new_inputs

    def gen_data(self):
        """
        to make the data which model need
        :return:
        """

        # 1，read the source text
        inputs, labels = self.read_data()
        print("read finished")

        # 2. word label index
        word_to_index, label_to_index = self.gen_word_label_index()
        # print("word_to_index",word_to_index)
        # print("label_to_index",label_to_index)
        print("vocab process finished")
        #
        # 3，input to index
        inputs_idx = self.trans_to_index(inputs, word_to_index)
        # print((inputs_idx))
        print("index transform finished")
        #
        # 4, padding
        inputs_idx = self.padding(inputs_idx, self._sequence_length)
        # print(inputs_idx)
        print("padding finished")

        # 6，label to index
        labels_idx = self.trans_label_to_index(labels, label_to_index)
        print("label index transform finished")

        return np.array(inputs_idx), np.array(labels_idx)

    def next_batch(self, x, y, batch_size):
        """
        生成batch数据集
        :param x: 输入
        :param y: 标签
        :param batch_size: 批量的大小
        :return:
        """
        perm = np.arange(len(x))
        np.random.shuffle(perm)
        x = x[perm]
        y = y[perm]

        num_batches = len(x) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = np.array(x[start: end], dtype="int64")
            batch_y = np.array(y[start: end], dtype="float32")

            yield dict(x=batch_x, y=batch_y)
