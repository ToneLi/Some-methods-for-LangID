# -*- coding: utf-8 -*-
class Parameters(object):

    word_embedding_dim = 300     #dimension of word embedding
    word_pre_training = None       #use vector_char trained by word2vec
    word_seq_length = 700         #max length of sentence
    num_classes = 235  #number of labels
    word_hidden_dim = 300     #the number of hidden units
    word_attention_size =300  #the size of attention layer
    keep_prob = 0.5        #droppout
    learning_rate = 1e-4  #learning rate
    lr_decay = 0.9          #learning rate decay
    clip = 5.0              #gradient clipping threshold

    num_epochs = 1  #epochs
    batch_size = 8    #batch_size

    word_vocab_size=None
    selfentity_vocab_size=None
