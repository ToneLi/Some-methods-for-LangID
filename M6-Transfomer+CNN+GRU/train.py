#encoding=utf-8
import json
import os
import datetime
import tensorflow as tf
from metrics import get_multi_metrics#,mean
now_path=os.getcwd()

from train_data_ import TrainData
from eval_data import EvalData
from  Transformer_CR import  TransformerCNNGRU
import numpy as np

class Trainer():
    def __init__(self, args):
        self.args = args
        with open(os.path.join(now_path, args), "r") as fr:
            self.config = json.load(fr) # the parameters about transformer


        self.train_data_o= None
        self.eval_data_o= None
        self.model = None
        # self.builder = tf.saved_model.builder.SavedModelBuilder("./pb_model/savedModel")

        #reload the datasets
        # print(TrainData.comeon(self.config))
        self.load_data()
        print("----------initialize the load data function---------")
        #
        self.train_inputs, self.train_labels, label_to_idx=self.train_data_o.gen_data()

        print("---over gen_data-----")
        print("label_to_idx",label_to_idx) #label_to_idx {'1': 0, '0': 1}

        # #
        print("train data size: {}".format(len(self.train_labels)))
        self.vocab_size = self.train_data_o.vocab_size
        print("vocab size: {}".format(self.vocab_size))
        # self.word_vectors = self.train_data_o.word_vectors
        self.label_list = [value for key, value in label_to_idx.items()]
        #
        self.eval_inputs, self.eval_labels = self.eval_data_obj.gen_data()
        print("eval data size: {}".format(len(self.eval_labels)))
        print("label numbers: ", len(self.label_list))
        # # 初始化模型对象
        self.create_model()
        print("------初始化结束--------")


    def load_data(self):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据
        self.train_data_o= TrainData(self.config)

        # 生成验证集对象和验证集数据
        self.eval_data_obj = EvalData(self.config)
    #
    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        self.model = TransformerCNNGRU(config=self.config, vocab_size=self.vocab_size)

    def sequence(self,x_batch):
        seq_len = []
        for line in x_batch:
            length = np.sum(np.sign(line))  # sign 输入：[3, 3, 0, 2.3]   输出：[1, 1,0,1]
            seq_len.append(length)
        return seq_len

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)#指定了每个gpu进程中使用显存的上限
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True,gpu_options=gpu_options)#当运行设备不满足要求时，会自动分配GPU或者CPU  ], gpu_options=gpu_options
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())
            current_step = 0

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(self.config["output_path"] + "/summary/train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)
    # #
            eval_summary_path = os.path.join(self.config["output_path"] + "/summary/eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)
    # #
            for epoch in range(self.config["epochs"]):
                print("----- Epoch {}/{} -----".format(epoch + 1, self.config["epochs"]))
    #
                for batch in self.train_data_o.next_batch(self.train_inputs, self.train_labels,
                                                            self.config["batch_size"]):
                    word_seq_len = self.sequence(batch["x"])

                    summary, loss, predictions = self.model.train(sess, batch, word_seq_len,self.config["keep_prob"])
                    # print(loss)
                    train_summary_writer.add_summary(summary)

                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["y"],
                                                                  labels=self.label_list)

                    print("-----loss-------:",loss,"-------acc-----:",acc)
                    # print("train: step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                    #     current_step, loss, acc, recall, prec, f_beta))
    #
                    current_step += 1
                    if self.eval_data_obj and current_step % self.config["checkpoint_every"] == 0:
                        # print("-----------我要开始验证了-------------------")

                        # eval_losses = []
                        # eval_accs = []
                        # eval_aucs = []
                        # eval_recalls = []
                        # eval_precs = []
                        # eval_f_betas = []
                        # for eval_batch in self.eval_data_obj.next_batch(self.eval_inputs, self.eval_labels,
                        #                                                 self.config["batch_size"]):
                        #     eval_summary, eval_loss, eval_predictions = self.model.eval(sess, self.config["sequence_length"], eval_batch)
                        #     eval_summary_writer.add_summary(eval_summary)
                        #
                        #     eval_losses.append(eval_loss)
                        #
                        #
                        #     acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                        #                                                   true_y=eval_batch["y"],
                        #                                                   labels=self.label_list)
                        #     eval_accs.append(acc)
                        #     eval_recalls.append(recall)
                        #     eval_precs.append(prec)
                        #     eval_f_betas.append(f_beta)
                        #
                        # print("\n")
                        # print("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        #     mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                        #     mean(eval_precs), mean(eval_f_betas)))
                        # print("\n")

                        if self.config["ckpt_model_path"]:
                            save_path = self.config["ckpt_model_path"]
                            if not os.path.exists(save_path):
                                os.makedirs(save_path)
                            model_save_path = os.path.join(save_path, self.config["model_name"])
                            self.model.saver.save(sess, model_save_path, global_step=current_step)





if __name__ == "__main__":
    begin = datetime.datetime.now()
    args="transformerCR_config.json"
    trainer = Trainer(args)
    trainer.train()
    end = datetime.datetime.now()
    print("------The run time is----------:", (end - begin).seconds)
