"""
time: 2018/12/13
title: deep learning in Emotional with keras by using Deep Learning
author: yuyuan's husband
case of terrible accuracy in test and validation ,we attempt to use binary category in sentiment.
"""
from __future__ import print_function
from ml_classify import get_data
from ml_classify import prepare_train_data
from ml_classify import build_vecs_tencent
from ml_classify import get_stop_word
from ml_classify import load_data
from ml_classify import confusion_matrix,plot_confusion_matrix
import re
import jieba
import json
from keras.preprocessing import sequence
from keras import backend as K
K.set_image_dim_ordering('th')
from keras.callbacks import Callback
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from keras.models import Sequential
from sklearn import metrics
from keras import backend as K
from keras.layers import Convolution2D, MaxPooling2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Model
from keras.layers import *
import numpy as np
from keras.engine.topology import Layer
from keras.utils import np_utils
import matplotlib.pyplot as plt

# from Yuan_postgraduate_SVMorBayes import confusion_matrix,plot_confusion_matrix

# class Metrics(Callback):
#     def on_train_begin(self, logs={}):
#         self.val_f1s = []
#         self.val_recalls = []
#         self.val_precisions = []
#
#     def on_epoch_end(self, epoch, logs={}):
#         val_predict = (np.asarray(self.model.predict(self.model.validation_data[0]))).round()
#         val_targ = self.model.validation_data[1]
#         _val_f1 = f1_score(val_targ, val_predict)
#         _val_recall = recall_score(val_targ, val_predict)
#         _val_precision = precision_score(val_targ, val_predict)
#         self.val_f1s.append(_val_f1)
#         self.val_recalls.append(_val_recall)
#         self.val_precisions.append(_val_precision)
#         print('-val_f1: %.4f --val_precision: %.4f --val_recall: %.4f' % (_val_f1, _val_precision, _val_recall))
#         return
#
# metrics = Metrics()

def Precision(y_true, y_pred):
    """精确率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))  # predicted positives
    precision = tp / (pp + K.epsilon())
    return precision

def Recall(y_true, y_pred):
    """召回率"""
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))  # true positives
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))  # possible positives
    recall = tp / (pp + K.epsilon())
    return recall

def F1(y_true, y_pred):
    """F1-score"""
    precision = Precision(y_true, y_pred)
    recall = Recall(y_true, y_pred)
    f1 = 2 * ((precision * recall) / (precision + recall + K.epsilon()))
    return f1

class Position_Embedding(Layer):
    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size == None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000., 2 * K.arange(self.size / 2, dtype='float32') / self.size)
        position_j = K.expand_dims(position_j, 0)
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1  # K.arange不支持变长，只好用这种方法生成
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate([K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)

class Attention(Layer):

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head * size_per_head
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        if seq_len == None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # 如果只传入Q_seq,K_seq,V_seq，那么就不做Mask
        # 如果同时传入Q_seq,K_seq,V_seq,Q_len,V_len，那么对多余部分做Mask
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # 对Q、K、V做线性变换
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # 计算内积，然后mask，然后softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # 输出并mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)

def handle_data(data):
    # special words need to be sub
    Normalization_data = []
    for dat in data:
        rule_date = '((\d+)年(\d+)月(\d+)日)|((\d+)年(\d+)月)|((\d+)月(\d+)日)|(\d+)年|(\d+)月|(\d+)日|(\d+)号'
        pattern2 = re.sub(rule_date, '', dat)
        rule_number = '(\d+)'
        res = re.sub(rule_number, '', pattern2)
        Normalization_data.append(res)
    return Normalization_data

def prepare_data(stop_word_list):
    pos_list, neutral_list, neg_list = get_data()
    # Normalization data
    pos_list = handle_data(pos_list)
    neutral_list = handle_data(neutral_list)
    neg_list = handle_data(neg_list)
    def get_seg_list(data):
        res_list = []
        for dat in data:
            seg_list = jieba.cut(dat, cut_all = False)
            seg_ = [seg for seg in seg_list if seg not in stop_word_list and seg != ' ']
            res_list.append(seg_)
        return res_list
    pos_list1 = get_seg_list(pos_list)
    neutral_list1 = get_seg_list(neutral_list)
    neg_list1 = get_seg_list(neg_list)
    return pos_list1, neutral_list1, neg_list1

def transform_to_matrix(x, padding_size, vec_size):
    json_dict = json.loads(open('tencent_word_embedding_dict.json', 'r', encoding='utf-8').read())
    res = []
    for sen in x:
        # print (sen)
        matrix = []
        for i in range(padding_size):
            try:
                array_ = json_dict[sen[i]]
                # print (array_)
                if array_ != 0:
                    array_list = array_
                    matrix.append(array_list)
                else:
                    # print (1)
                    matrix.append([0] * vec_size)
            except Exception as e:
            # 这里有两种except情况，
            # 1. 这个单词找不到
            # 2. sen没那么长
            # 不管哪种情况，我们直接贴上全是0的vec
            #     print("Error type",e)
                matrix.append([0] * vec_size)
        res.append(matrix)
    return res

def load_embedding_data():
    # 对于输入序列先不做停用处理，要不容易padding都是0，试一下numpy（sample,40，200）的卷积效果？,
    # 转换为 numpy（sample,200,40）这样的形式？
    # stop_word_list = get_stop_word()
    #先设为[]
    stop_word_list = []
    pos_list, neutral_list, neg_list = prepare_data(stop_word_list)
    # print (pos_list[0])
    # print (len(pos_list))
    # pos_list, neg_list = prepare_data(stop_word_list)
    # 用vector表示出一个大matrix，并用CNN做“降维+注意力”
    # 因为处理的是短文本的微博数据，所以我们只选取前128个分词的结果作为padding_size
    # 所以维度为【202,128,200】
    pos_input_data = transform_to_matrix(pos_list, 40, 200)
    # print ("pos_input_data",pos_input_data[0])
    # print (len(pos_input_data))
    neutral_input_data = transform_to_matrix(neutral_list, 40, 200)
    neg_input_data = transform_to_matrix(neg_list, 40, 200)
    y = np.concatenate((np.ones(len(pos_input_data)) * 2, np.zeros(len(neg_input_data)), np.ones(len(neutral_input_data))))
    # y = np.concatenate((np.ones(len(pos_input_data)) * 2, np.zeros(len(neg_input_data))))
    X = pos_input_data[:]
    for neg in neg_input_data:
        X.append(neg)
    for neu in neutral_input_data:
        X.append(neu)
    train_X = np.array(X)
    # print (train_X.shape)
    # print (y.shape)
    train_ratio = 0.2
    train_x, test_x, train_y, test_y = prepare_train_data(train_X, y, train_ratio)
    return train_x, test_x, train_y, test_y

def load_number_data():
    pos_list, neutral_list, neg_list = get_data()
    def get_num_dict():
        data = load_data()
        num_list = []
        for uuid in data:
            seg_list = jieba.cut(data[uuid][0], cut_all = False)
            for seg in seg_list:
                if seg not in num_list:
                    num_list.append(seg)
        print ('Total seg num: ', len(num_list))
        num_dict = dict(enumerate(num_list))
        res_num_dict = {}
        for k,v in num_dict.items():
            res_num_dict[v] = k
        return res_num_dict
    num_dict = get_num_dict()
    def get_seg_list(pos_list, neutral_list, neg_list, num_dict):
        pos_seg_list = []
        neutral_seg_list = []
        neg_seg_list = []
        for p in pos_list:
            seg_list = jieba.cut(p, cut_all=False)
            res = [num_dict[seg] for seg in seg_list]
            pos_seg_list.append(res)
        for p in neutral_list:
            seg_list = jieba.cut(p, cut_all=False)
            res = [num_dict[seg] for seg in seg_list]
            neutral_seg_list.append(res)
        for p in neg_list:
            seg_list = jieba.cut(p, cut_all=False)
            res = [num_dict[seg] for seg in seg_list]
            neg_seg_list.append(res)
        return pos_seg_list, neutral_seg_list, neg_seg_list
    pos_seg_list, neutral_seg_list, neg_seg_list = get_seg_list(pos_list, neutral_list, neg_list,num_dict)
    print (len(pos_seg_list))
    print (len(neutral_seg_list))
    print (len(neg_seg_list))
    # print (pos_seg_list[0])
    y = np.concatenate((np.ones(len(pos_seg_list)) * 2, np.zeros(len(neg_seg_list)), np.ones(len(neutral_seg_list))))
    X = pos_seg_list[:]
    for neg in neg_seg_list:
        X.append(neg)
    for neu in neutral_seg_list:
        X.append(neu)
    train_X = np.array(X)
    # print (train_X.shape)
    # print (y.shape)
    train_ratio = 0.2
    train_x, test_x, train_y, test_y = prepare_train_data(train_X, y, train_ratio)
    return train_x, test_x, train_y, test_y

def cnn_1D_model():
    train_x, test_x, train_y, test_y = load_embedding_data()
    X_train = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
    X_test = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])
    print(X_train.shape, X_test.shape)
    y_train = np_utils.to_categorical(train_y)
    y_test = np_utils.to_categorical(test_y)
    #set parameters:
    batch_size = 256
    n_filter = 32
    filter_length = 3
    nb_epoch = 10
    n_pool = 2
    model = Sequential()
    ### 终于找到问题了，因为keras的关于图片的维度顺序有两种类型，分别是“th”和”tf“，需要在开始定义‘th’
    model.add(Convolution1D(n_filter ,filter_length, filter_length, input_shape = (1, 40, 200)))
    model.add(Activation('relu'))
    model.add(Convolution2D(n_filter ,filter_length, filter_length))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(n_pool, n_pool)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    #后面接上一个ANN
    model.add(Dense(200))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])
    model.fit(X_train,y_train, batch_size = batch_size, epochs = nb_epoch, validation_data=(X_test, y_test),verbose = 2)
    # score = model.evaluate(X_test, test_y, verbose=0)
    # print ('Test score: ', score[0])
    # print ('Test accuracy: ', score[1])
    model.save('./Sentiment_models/CNN1D_model.h5')

class LeNet:
    @staticmethod
    def bulid(input_shape, classes):
        model = Sequential()
        #第一层滤波器加最大池化
        #padding分为两种方式，same和valid，如果max_pooling的时候，窗口不够，那么same会添加一列0来进行滑动
        model.add(Convolution2D(64,(3,3), padding = 'same', input_shape = input_shape))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
        #第二层卷积层中增加滤波器的个数。
        model.add(Conv2D(128, (3,3), border_mode = 'same'))
        model.add(Activation('relu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
        #然后添加全连接网络
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))
        model.add(Dropout(0.25))
        model.add(Dense(3))
        model.add(Dense(classes))
        model.add(Activation('softmax'))
        return model

def cnn_2D_model():
    epoch = 100
    batch_size = 128
    verbose = 1
    classes = 3
    hidden_size = 128
    validation_split = 0.2
    dropout = .3
    # 每列是一个词向量与每行是一个词向量进行卷积对比，
    input_shape = (1, 40, 200)
    train_x, test_x, train_y, test_y = load_embedding_data()
    # print (train_x.shape)
    # x = train_x.transpose()
    # print (x.shape)
    X_train = train_x.reshape(train_x.shape[0], 1, train_x.shape[1], train_x.shape[2])
    X_test = test_x.reshape(test_x.shape[0], 1, test_x.shape[1], test_x.shape[2])
    # print(X_train.shape, X_test.shape)
    x_train = X_train.astype('float')
    x_test = X_test.astype('float')
    y_train = np_utils.to_categorical(train_y, classes)
    y_test = np_utils.to_categorical(test_y, classes)
    model = LeNet.bulid(input_shape=input_shape, classes = classes)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=verbose,
                        validation_split=validation_split)
    score = model.evaluate(x_test, y_test, verbose=verbose)
    print('CNN_model test loss:', score[0])
    print('CNN_model test accuracy;', score[1])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('CNN_model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = 'upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('CNN_model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('./Sentiment_models/CNN_model.h5')

def MLP_model():

    epoch = 100
    batch_size = 128
    verbose = 2
    classes = 3
    hidden_size = 128
    validation_split = 0.2
    dropout = .3
    # pos_list, neutral_list, neg_list = get_data()
    # train_x, y = build_vecs_tencent(pos_list, neutral_list, neg_list)
    # train_x, test_x, train_y, test_y = prepare_train_data(train_x, y, 0.2)
    # print (train_x.shape)
    train_x, test_x, train_y, test_y = json.load(open('train_x.json', 'r'))
    train_x = np.array(train_x)
    test_x = np.array(test_x)
    train_y = np.array(train_y)
    test_y = np.array(test_y)
    x_train = train_x.astype('float')
    x_test = test_x.astype('float')
    y_train = np_utils.to_categorical(train_y, classes)
    y_test = np_utils.to_categorical(test_y, classes)
    model = Sequential()
    model.add(Dense(hidden_size, input_shape=(200,)))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(hidden_size))
    model.add(Activation('relu'))
    model.add(Dropout(dropout))
    model.add(Dense(classes))
    model.add(Activation('softmax'))
    # model.summary()
    model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy',
                                                                                Precision, Recall, F1])
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epoch, verbose=verbose,
                        validation_data=(x_test, y_test))
    # model.summary()
    score = model.evaluate(x_test, y_test, verbose=verbose)
    print('bp_model test loss:', score[0])
    print('bp_model test accuracy;', score[1])
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('MLP_model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('MLP_model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('./Sentiment_models/BP_model.h5')

    # pred = model.predict(x_test)

    # classify_report = metrics.classification_report(y_test, pred)
    # print('classify_report', classify_report)

    # cnf_matrix = confusion_matrix(y_test, pred)
    # class_names = [0, 1, 2]
    # plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='MLP Confusion matrix')
    # plt.show()

def lstm_model():
    max_sentence_length = 80
    max_features = 20000
    batch_size = 32
    embedding_size = 128
    hidden_layer_size = 64
    train_x, test_x, train_y, test_y = load_number_data()
    # print('train_x.shape',train_x.shape)
    y_train = np_utils.to_categorical(train_y)
    y_test = np_utils.to_categorical(test_y)
    x_train = sequence.pad_sequences(train_x, maxlen=max_sentence_length)
    x_test = sequence.pad_sequences(test_x, maxlen=max_sentence_length)
    # print ('x_train:',x_train)
    model = Sequential()
    # 这里用的是keras自带的embedding层来进行词向量,考虑进行替换成腾讯词向量.
    # embedding_layer = Embedding(num_words,
    #                             EMBEDDING_DIM,
    #                             weights=[embedding_matrix],
    #                             input_length=MAX_SEQUENCE_LENGTH,
    #                             trainable=False)
    model.add(Embedding(max_features, embedding_size, input_length = max_sentence_length))
    # model.add(SpatialDropout1D(Dropout(.2)))
    # model.add(LSTM(hidden_layer_size, dropout = 0.2, recurrent_dropout = 0.2))
    model.add(Bidirectional(CuDNNLSTM(hidden_layer_size)))
    model.add(Dense(3))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    model.summary()
    history = model.fit(x_train, y_train, batch_size=batch_size, epochs=100, validation_data=(x_test, y_test))
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('LSTM_model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('LSTM_model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    model.save('./Sentiment_models/Bi-LSTM_model.h5')

def attention_keras():
    train_x, test_x, train_y, test_y = load_number_data()
    # print (len(train_x))
    # print (test_x.shape)
    # print (test_y.shape)
    # y_train = train_y
    # y_test = test_y
    # 如果使用categorical_crossentropy这个损失函数的时候，y的输入需要变成类别的one - hot编码形式
    y_train = np_utils.to_categorical(train_y)
    y_test = np_utils.to_categorical(test_y)
    # print (y_train.shape)
    # print (y_train[1])
    max_features = 20000
    maxlen = 80
    batch_size = 128
    x_train = sequence.pad_sequences(train_x, maxlen=maxlen)
    x_test = sequence.pad_sequences(test_x, maxlen=maxlen)
    S_inputs = Input(shape=(None,), dtype='int32')
    embeddings = Embedding(max_features, 128)(S_inputs)
    embeddings = Position_Embedding()(embeddings)
    O_seq = Attention(8, 16)([embeddings, embeddings, embeddings])
    O_seq = GlobalAveragePooling1D()(O_seq)
    O_seq = Dropout(0.25)(O_seq)
    # outputs = Dense(1, activation='sigmoid')(O_seq)
    ### 用了loss='categorical_crossentropy'之后，output层的输出也需要依次的进行修改，输出就不是单个值了，这里需要改为我们的分类个数
    outputs = Dense(3, activation='softmax')(O_seq)
    model = Model(inputs=S_inputs, outputs=outputs)
    # model summarization
    model.summary()
    # try using different optimizers and different optimizer configs
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    print('Train...')
    # 默认verbose = 1
    # model.fit(x_train, y_train,
    #           batch_size=batch_size,
    #           epochs=10,
    #           validation_data=(x_test, y_test),
    #         verbose = 2
    #           )
    ###下面来试一下用图来表示训练过程中的损失
    history = model.fit(x_train, y_train,batch_size = batch_size,epochs = 100, validation_data=(x_test, y_test))
    # plt.subplot(211)
    # plt.title("Accuracy")
    # plt.plot(history.history['acc'], color = 'g', label = 'Train')
    # plt.plot(history.history['val_acc'], color = 'b', label = 'Validation')
    # plt.legend(loc = 'best')
    #
    # plt.subplot(212)
    # plt.title("Loss")
    # plt.plot(history.history['loss'], color='g', label='Train')
    # plt.plot(history.history['val_loss'], color='b', label='Validation')
    # plt.legend(loc='best')

    # plt.tight_layout()
    # plt.show()
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Attention_model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Attention_model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    model.save('./Sentiment_models/Attention_model.h5')

if __name__ == "__main__":
    # cnn_1D_model()
    # 2DCNN 的准确率在92%
    # cnn_2D_model()
    # MLP 模型的准确率在89%
    MLP_model()
    # lstm模型的准确率在91.67%
    # lstm_model()
    # attention的准确率在92%
    # attention_keras()


