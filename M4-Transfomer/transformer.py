#encoding=utf-8
import tensorflow as tf
import numpy as np
from base import BaseModel

class TransformerModel(BaseModel):  #basemodel is the father
    def __init__(self, config, vocab_size): #__init__(self, config, vocab_size, word_vectors):
        super(TransformerModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=None)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.contrib.layers.xavier_initializer()) #return 返回值：初始化权重矩阵

            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)


        with tf.name_scope("positionEmbedding"):
            embedded_position = self._position_embedding()
            print(embedded_position)  # shape=(128, 700, 200) （batch_size, sentence_length,embedding_size）
            print("----------embedded_position-------over-----")
    #
        embedded_representation = embedded_words + embedded_position   ## embedded_representation Tensor("add:0", shape=(128, 700, 200), dtype=float32) （batch_size, sentence_length,embedding_size）

        with tf.name_scope("transformer"):
            for i in range(self.config["num_blocks"]):
                with tf.name_scope("transformer-{}".format(i + 1)):
                    with tf.name_scope("multi_head_attention"):
                        # 维度[batch_size, sequence_length, embedding_size]
                        multihead_atten = self._multihead_attention(inputs=self.inputs,
                                                                    queries=embedded_representation,
                                                                    keys=embedded_representation)
                        embedded_representation = multihead_atten
    #
            outputs = tf.reshape(embedded_representation,
                                 [-1, self.config["sequence_length"] * self.config["embedding_size"]])
            # print("output",outputs)
        output_size = outputs.get_shape()[-1].value

        with tf.name_scope("dropout"):
            outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "output_w",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(outputs, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        # 计算交叉熵损失
        self.loss = self.cal_loss() + self.config["l2_reg_lambda"] * self.l2_loss
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()


    def _layer_normalization(self, inputs):
        """
        对最后维度的结果做归一化，也就是说对每个样本每个时间步输出的向量做归一化
        :param inputs:
        :return:
        """
        epsilon = self.config["ln_epsilon"]

        inputs_shape = inputs.get_shape()  # [batch_size, sequence_length, embedding_size]
        params_shape = inputs_shape[-1]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)  ## 用于在指定维度计算均值与方差

        beta = tf.Variable(tf.zeros(params_shape), dtype=tf.float32)

        gamma = tf.Variable(tf.ones(params_shape), dtype=tf.float32)
        normalized = (inputs - mean) / ((variance + epsilon) ** .5)

        outputs = gamma * normalized + beta

        return outputs
    #
    def _multihead_attention(self, inputs, queries, keys, num_units=None):
        """
        计算多头注意力
        :param inputs: 原始输入，用于计算mask
        :param queries: 添加了位置向量的词向量
        :param keys: 添加了位置向量的词向量
        :param num_units: 计算多头注意力后的向量长度，如果为None，则取embedding_size
        :return:
        """
        num_heads = self.config["num_heads"]  # multi head 的头数

        if num_units is None:  # 若是没传入值，直接去输入数据的最后一维，即embedding size.
            num_units = queries.get_shape().as_list()[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu) #和 queries  维度一样（前提 num_units is None， 为 shape=(128, 700, 200) （batch_size, sentence_length,embedding_size）
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        # 将数据按最后一维分割成num_heads个, 然后按照第一维拼接
        # Q, K, V 的维度都是[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)#eg:(1024, 700, 25)

        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)#shape=(1024, 700, 25)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)
    #
        # 计算keys和queries之间的点积，维度[batch_size * numHeads, queries_len, key_len], 后两维是queries和keys的序列长度
        similarity = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))#(1):tf.transpose(K_, [0, 2, 1])=shape=(1024, 25, 100) (2)similarity_shape=(1024, 700, 700)

        # 对计算的点积进行缩放处理，除以向量长度的根号值
        similarity = similarity / (K_.get_shape().as_list()[-1] ** 0.5)  #Tensor("transformer/transformer-1/multi_head_attention/truediv:0", shape=(1024, 700, 700), dtype=float32)
        mask = tf.tile(inputs, [num_heads, 1])
        print("mask",mask)
        #
        # 增加一个维度，并进行扩张，得到维度[batch_size * numHeads, queries_len, keys_len]
        key_masks = tf.tile(tf.expand_dims(mask, 1), [1, tf.shape(queries)[1], 1])
        #
        # # tf.ones_like生成元素全为1，维度和similarity相同的tensor, 然后得到负无穷大的值
        paddings = tf.ones_like(similarity) * (-2 ** 32 + 1)
        #
        # tf.where(condition, x, y),condition中的元素为bool值，其中对应的True用x中的元素替换，对应的False用y中的元素替换
        # 因此condition,x,y的维度是一样的。下面就是keyMasks中的值为0就用paddings中的值替换
        masked_similarity = tf.where(tf.equal(key_masks, 0), paddings,
                                     similarity)  # 维度[batch_size * numHeads, queries_len, key_len]   和  key_masks  维度一样
        #
        # 通过softmax计算权重系数，维度 [batch_size * numHeads, queries_len, keys_len]
        weights = tf.nn.softmax(masked_similarity)

        query_masks = tf.tile(tf.expand_dims(mask, -1), [1, 1, tf.shape(keys)[1]])
        mask_weights = tf.where(tf.equal(query_masks, 0), paddings,
                                weights)  # 维度[batch_size * numHeads, queries_len, key_len]
        #
        # 加权和得到输出值, 维度[batch_size * numHeads, sequence_length, embedding_size/numHeads]
        outputs = tf.matmul(mask_weights, V_)
        #
        # 将多头Attention计算的得到的输出重组成最初的维度[batch_size, sequence_length, embedding_size]
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)

        outputs = tf.nn.dropout(outputs, keep_prob=self.keep_prob)

        # 对每个subLayers建立残差连接，即H(x) = F(x) + x
        outputs += queries
        # normalization 层
        outputs = self._layer_normalization(outputs)
        return outputs

    def _position_embedding(self):
        """
        生成位置向量, to get the position embedding
        :return:
        """
        batch_size = self.config["batch_size"]
        sequence_length = self.config["sequence_length"]
        embedding_size = self.config["embedding_size"]

        # 生成位置的索引，并扩张到batch中所有的样本上
        position_index = tf.tile(tf.expand_dims(tf.range(sequence_length), 0), [batch_size, 1])

        # 根据正弦和余弦函数来获得每个位置上的embedding的第一部分
        position_embedding = np.array([[pos / np.power(10000, (i - i % 2) / embedding_size) for i in range(embedding_size)] for pos in range(sequence_length)])

        # 然后根据奇偶性分别用sin和cos函数来包装
        position_embedding[:, 0::2] = np.sin(position_embedding[:, 0::2])  #偶数  从位置为0开始  步数是2
        position_embedding[:, 1::2] = np.cos(position_embedding[:, 1::2])#奇数  从位置为1开始  步数是2

        # 将positionEmbedding转换成tensor的格式
        position_embedding = tf.cast(position_embedding, dtype=tf.float32)

        # 得到三维的矩阵[batchSize, sequenceLen, embeddingSize]
        embedded_position = tf.nn.embedding_lookup(position_embedding, position_index)

        return embedded_position
