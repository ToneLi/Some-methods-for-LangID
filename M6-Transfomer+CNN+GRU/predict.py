#encoding=utf-8
import os
import json
import datetime
import tensorflow as tf
from  Transformer_CR import  TransformerCNNGRU
from eval_data import EvalData
from metrics import get_multi_metrics,mean
import numpy as np
now_path=os.getcwd()

class Predictor():
    def __init__(self, args):
        self.args = args
        with open(os.path.join(now_path, args), "r") as fr:
            config = json.load(fr)  #


        self.model = None
        self.config = config
        self._dic_path = config["char_dic"]
        self._label_list_path = config["label_list"]
        self.word_to_index, self.label_to_index = self.gen_word_label_index()
        self.index_to_label = {value: key for key, value in self.label_to_index.items()}
        self.vocab_size = len(self.word_to_index)
        self.word_vectors = None
        self.sequence_length = self.config["sequence_length"]
        self.eval_data_obj = EvalData(self.config)
        # 创建模型
        self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        self.create_model()
        # 加载计算图
        self.load_graph()


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
                    line1 = line1.strip().split("\t")
                    labels.append(line1[0])

                except:
                    continue

        label_to_index = dict(zip(labels, list(range(len(labels)))))

        return word_to_index, label_to_index


    def sequence(self,x_batch):
        seq_len = []
        for line in x_batch:
            length = np.sum(np.sign(line))  # sign 输入：[3, 3, 0, 2.3]   输出：[1, 1,0,1]
            seq_len.append(length)
        return seq_len


    def sentence_to_idx(self, sentence):
        """
        将分词后的句子转换成idx表示
        :param sentence:
        :return:
        """
        sentence_ids = [self.word_to_index.get(token, self.word_to_index["<UNK>"]) for token in sentence]
        sentence_pad = sentence_ids[: self.sequence_length] if len(sentence_ids) > self.sequence_length \
            else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))
        return sentence_pad


    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.config["ckpt_model_path"])
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.config["ckpt_model_path"]))


    def create_model(self):
        self.model = TransformerCNNGRU(config=self.config, vocab_size=self.vocab_size)



    def predict(self):
        eval_accs = []
        label_list=[]
        with open(self._label_list_path,"r",encoding="utf-8") as fr:
            for line in fr.readlines():
                label_list.append(line.strip())


        for eval_batch in self.eval_data_obj.next_batch(self.eval_inputs, self.eval_labels,
                                                        self.config["batch_size"]):
            word_seq_len = self.sequence(eval_batch["x"])
            # print(word_seq_len)
            # print(eval_batch["x"])
            eval_summary, eval_loss, eval_predictions = self.model.eval(self.sess, eval_batch,word_seq_len)
            # print(eval_loss)
        # #
            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                          true_y=eval_batch["y"],
                                                          labels=label_list)
            print("准确度是-----",acc)
            eval_accs.append(acc)
        #
        print("--------the accuracy is--------:",mean(eval_accs))


if __name__=="__main__":
    begin=datetime.datetime.now()
    args = "transformerCR_config.json"
    predict=Predictor(args=args)
    predict.predict()
    end=datetime.datetime.now()
    print("------The run time is----------:",(end-begin).seconds)