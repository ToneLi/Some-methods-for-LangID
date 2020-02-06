#encoding=utf-8
import tensorflow as tf
from Parameters import Parameters as pm
from data_processing import *
from tensorflow.contrib import layers
class Rnn_Attention(object):

    def __init__(self):
        """---输入-----------"""
        # word level
        self.input_word_x1 = tf.placeholder(tf.int32, shape=[None, pm.word_seq_length], name='input_word_x1')
        self.input_y1 = tf.placeholder(tf.float32, shape=[None, pm.num_classes], name='input_y1')


        """------句子长度-----------"""
        self.word_seq_length = tf.placeholder(tf.int32, shape=[None], name='word_seq_length')


        # 2 type
        self.keep_pro = tf.placeholder(tf.float32, name='drop_out')
        self.global_step = tf.Variable(0, trainable=False, name='global_step')
        self.Rnn_attention()


    def Rnn_attention(self):
        with tf.variable_scope('word_Cell'): #word 细胞
            word_cell_fw = tf.contrib.rnn.GRUCell(pm.word_hidden_dim)
            word_Cell_fw = tf.contrib.rnn.DropoutWrapper(word_cell_fw, self.keep_pro)
            word_cell_bw = tf.contrib.rnn.GRUCell(pm.word_hidden_dim)
            word_Cell_bw = tf.contrib.rnn.DropoutWrapper(word_cell_bw, self.keep_pro)


        """----------输入嵌入矩阵-----------"""
        #word 水平
        with tf.variable_scope('word_embedding'):
            self.embedding = tf.Variable(tf.random_uniform([pm.word_vocab_size, pm.word_embedding_dim], -1.0, 1.0),name="W")

            self.embedding_input = tf.nn.embedding_lookup(self.embedding, self.input_word_x1)
            # print(self.embedding)


        with tf.variable_scope('biRNN'):
            wordoutput, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw=word_Cell_fw, cell_bw=word_Cell_bw, inputs=self.embedding_input,
                                                        sequence_length=self.word_seq_length, dtype=tf.float32)
            wordoutput = tf.concat(wordoutput, 2) #[batch_size, seq_length, 2*hidden_dim] shape=(?, 34, 100),

            # print("wordout",wordoutput)

            # print(".................",selfentity_output)

        with tf.variable_scope('word_attention'):
            u_list = []
            seq_size = wordoutput.shape[1].value
            hidden_size = wordoutput.shape[2].value #[2*hidden_dim]
            # print("hidden_size:",hidden_size)
            word_attention_w = tf.Variable(tf.truncated_normal([hidden_size, pm.word_attention_size], stddev=0.1), name='attention_w')#stddev=0.05
            word_attention_u = tf.Variable(tf.truncated_normal([pm.word_attention_size, 1], stddev=0.1), name='attention_u')
            word_attention_b = tf.Variable(tf.constant(0.1, shape=[pm.word_attention_size]), name='attention_b')
            word_attention_V=tf.Variable(tf.constant(0.1, shape=[pm.word_attention_size,pm.word_attention_size]), name='attention_b')

            for t in range(seq_size):
                #u_t:[1,attention]

                # print(wordoutput)#shape=(?, 34, 100)
                # print(wordoutput[:, t, :])#shape=(?, 100)
                u_t = tf.tanh(tf.matmul(wordoutput[:, t, :], word_attention_w) + tf.reshape(word_attention_b, [1, -1]))#tf.reshape(tensor,[-1,1])将张量变为一维列向量tf.reshape(tensor,[1,-1])将张量变为一维行向量
                # print("--------你猜是神魔------",tf.matmul(wordoutput[:, t, :], word_attention_w))
                u_t=tf.matmul(u_t,tf.transpose(word_attention_V))
                u = tf.matmul(u_t, word_attention_u)
                u_list.append(u)
            logit = tf.concat(u_list, axis=1)
            #u[seq_size:attention_z]
            weights = tf.nn.softmax(logit, name='attention_weights')
            #weight:[seq_size:1]
            # print("weight",weights)#shape=(?, 34)
            # print((tf.reshape(weights, [-1, seq_size, 1]), 1))#shape=(?, 34, 1)
            word_out_final = tf.reduce_sum(wordoutput * tf.reshape(weights, [-1, seq_size, 1]), 1)
            # print("word_out_final",word_out_final)

        # print("end_final",end_final)

        with tf.name_scope('dropout'):
            self.out_drop = tf.nn.dropout(word_out_final, keep_prob=self.keep_pro)

        with tf.name_scope('output'):
            w = tf.Variable(tf.truncated_normal([2*(pm.word_hidden_dim), pm.num_classes], stddev=0.1), name='w')
            b = tf.Variable(tf.zeros([pm.num_classes]), name='b')
            self.logits = tf.matmul(self.out_drop, w) + b
            self.predict = tf.argmax(tf.nn.softmax(self.logits), 1, name='predict') #shape=(?,),

        with tf.name_scope('loss'):
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y1)
            self.loss = tf.reduce_mean(cross_entropy)#+layers.l1_regularizer(0.7)(w)+layers.l2_regularizer(0.3)(w)
    #
        with tf.name_scope('optimizer'):
            # 退化学习率 learning_rate = lr*(0.9**(global_step/10);staircase=True表示每decay_steps更新梯度
            # learning_rate = tf.train.exponential_decay(self.config.lr, global_step=self.global_step,
            # decay_steps=10, decay_rate=self.config.lr_decay, staircase=True)
            # optimizer = tf.train.AdamOptimizer(learning_rate)
            # self.optimizer = optimizer.minimize(self.loss, global_step=self.global_step) #global_step 自动+1
            # no.2
            optimizer = tf.train.AdamOptimizer(pm.learning_rate)
            gradients, variables = zip(*optimizer.compute_gradients(self.loss))  # 计算变量梯度，得到梯度倿变�?
            gradients, _ = tf.clip_by_global_norm(gradients, pm.clip)
            # 对g进行l2正则化计算，比较其与clip的值，如果l2后的值更大，让梯�?clip/l2_g),得到新梯�?
            self.optimizer = optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)
            # global_step 自动+1
    #
        with tf.name_scope('accuracy'):
            correct = tf.equal(self.predict, tf.argmax(self.input_y1, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name='accuracy')
            # print(self.accuracy)


    def feed_data(self, word_x_batch, word_y_batch, word_seq_length,keep_pro):
        feed_dict = {self.input_word_x1: word_x_batch,
                    self.input_y1: word_y_batch,

                    self.word_seq_length: word_seq_length,

                    self.keep_pro: keep_pro}

        return feed_dict

    #

    def evaluate(self, sess, x1, y1):
        batch_test = batch_iter(x1, y1, batch_size=64)
        for x_batch, y_batch in batch_test:


            seq_len = sequence(x_batch)

            feed_dict = self.feed_data(x_batch, y_batch, seq_len, 1.0)
            test_loss, test_accuracy = sess.run([self.loss, self.accuracy], feed_dict=feed_dict)


        return test_loss, test_accuracy








