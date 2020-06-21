# coding: utf-8

import sys
import os
import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score
from tensorflow.contrib.layers.python.layers import batch_norm
from time import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



class DeepFm(object):
    def __init__(self, feature_num,
                 max_feature_length=20,
                 embedding_size=8,
                 deep_layers=[8, 8],
                 dropout_deep=[1.0, 1.0, 1.0],
                 dropout_fm=[1.0, 1.0],
                 learning_rate=0.01,
                 is_batch_norm=False,
                 batch_norm_decay=0.995,
                 deep_layer_activation=tf.nn.relu,
                 field_size=2,
                 batch_size=2,
                 l2_eta=0.0,
                 optimizer_type='adam',
                 display=200000,
                 feature_mean=False,
                 random_seed=2020
                 ):
        self.embedding_size = embedding_size
        self.max_feature_length = max_feature_length

        self.deep_layers = deep_layers
        self.dropout_deep = dropout_deep
        self.dropout_fm = dropout_fm

        self.learning_rate = learning_rate
        self.l2_eta = l2_eta
        self.deep_layers_activation = deep_layer_activation

        self.is_batch_norm = is_batch_norm
        self.batch_norm_decay = batch_norm_decay

        self.feature_num = feature_num
        self.field_size = field_size
        self.batch_size = batch_size

        self.optimizer_type = optimizer_type
        self.random_seed = random_seed
        self.display = display
        self.feature_mean = feature_mean
        self.weights = dict()
        self._init_weight()
        self._init_session()
        #self._init_graph()

    def load_data(self, file_name, epoch=1, buf_size=20480):
        data_set = tf.data.TextLineDataset(file_name)
        data_set = data_set.map(self.adress_data)
        padded_shapes = (tf.TensorShape([self.max_feature_length]),
                         tf.TensorShape([self.max_feature_length]),
                         tf.TensorShape([self.max_feature_length]),
                         tf.TensorShape([self.max_feature_length]),
                         tf.TensorShape([]))
        # data_set = data_set.repeat(epoch).padded_batch(self.batch_size, padded_shapes=padded_shapes)
        data_set = data_set.shuffle(buf_size).padded_batch(self.batch_size, padded_shapes=padded_shapes)
        data_set = data_set.repeat(epoch)
        data_iter = data_set.make_one_shot_iterator()
        q, d, q_v, d_v, train_label = data_iter.get_next()
        return q, d, q_v, d_v, train_label

    def _init_graph(self, q, d, q_v, d_v, train_label, is_train=True):
        # self.graph = tf.Graph()
        # with self.graph.as_default():

        #神经网络
        self.dropout_deep_layer = tf.constant(self.dropout_deep, dtype=tf.float32)
        self.dropout_fm_layer = tf.constant(self.dropout_fm, dtype=tf.float32)
        self.train_phrase = tf.constant(is_train, dtype=tf.bool)

        self.feat_index = tf.stack([q, d], axis=1)
        self.feat_value = tf.stack([q_v, d_v], axis=1)
        # self.train_label = tf.reshape(self.train_label, shape=[-1,1])
        print(self.feat_index)

        # embedding_lookup
        self.embeddings = tf.nn.embedding_lookup(self.weights['feature_embedding'], self.feat_index)
        self.reshaped_feat_value = tf.reshape(self.feat_value, shape=[-1, self.field_size, self.max_feature_length, 1])
        self.embeddings = tf.multiply(self.embeddings, self.reshaped_feat_value)

        # fm first
        self.fm_first = tf.nn.embedding_lookup(self.weights['feature_bias'], self.feat_index)
        self.fm_first = tf.reduce_sum(tf.multiply(self.fm_first, self.reshaped_feat_value), axis=[2])
        self.fm_first = tf.reshape(self.fm_first, shape=[-1, self.field_size])
        # self.fm_first = tf.nn.dropout(self.fm_first, self.dropout_fm_layer[0])

        # fm second
        self.sum_embedding = tf.reduce_sum(self.embeddings, axis=[1, 2])
        self.sum_embedding_square = tf.square(self.sum_embedding)

        self.square_embedding = tf.square(self.embeddings)
        self.square_embedding_sum = tf.reduce_sum(self.square_embedding, axis=[1, 2])

        self.fm_second = 0.5*tf.subtract(self.sum_embedding_square, self.square_embedding_sum)
        self.fm_second = tf.reshape(self.fm_second, shape=[-1, self.embedding_size])
        # self.fm_second = tf.nn.dropout(self.fm_second, self.dropout_fm_layer[1])

        # deep
        self.y_deep = tf.reshape(tf.reduce_sum(self.embeddings, axis=2),
                                 shape=[-1, self.field_size*self.embedding_size])
        # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_deep_layer[0])

        if self.feature_mean:
            self.not_zero_feature = tf.to_float(tf.count_nonzero(self.embeddings, axis=2))
            self.not_zero_feature = tf.reshape(self.not_zero_feature, shape=[-1, self.field_size*self.embedding_size])
            self.y_deep = tf.divide(self.y_deep, self.not_zero_feature)

        for i in range(len(self.deep_layers)):
            self.y_deep = tf.add(tf.matmul(self.y_deep, self.weights['layer_%d'%(i)]), self.weights['bias_%d'%(i)])
            if self.is_batch_norm:
                self.y_deep = self.batch_norm_layer(self.y_deep, train_phrase=self.train_phrase, scope_bn="bn_%d"%(i))
            self.y_deep = self.deep_layers_activation(self.y_deep)
            # self.y_deep = tf.nn.dropout(self.y_deep, self.dropout_deep_layer[i+1])

        #final_layer
        self.final_input = tf.concat([self.fm_first, self.fm_second, self.y_deep], axis=1)
        self.out = tf.nn.sigmoid(
            tf.add(tf.matmul(self.final_input, self.weights['layer_concat']), self.weights['bias_concat']))

        if not is_train:
            return self

        # loss
        self.loss = tf.losses.log_loss(tf.reshape(train_label, shape=[-1, 1]), self.out)
        if self.l2_eta > 0:
            self.loss += tf.contrib.layers.l2_regularizer(
                self.l2_eta)(self.weights['layer_concat'])
            for i in range(len(self.deep_layers)):
                self.loss += tf.contrib.layers.l2_regularizer(
                    self.l2_eta)(self.weights['layer_%d'%(i)])

        if self.optimizer_type == 'adam':
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                    beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'adagrad':
            self.optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate,
                                                       initial_accumulator_value=1e-8).minimize(self.loss)
        elif self.optimizer_type == 'gd':
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.loss)
        elif self.optimizer_type == 'momentum':
            self.optimizer = tf.train.MomentumOptimizer(learning_rate=self.learning_rate,
                                                        momentum=0.95).minimize(self.loss)

        return self

    def _init_weight(self):
        tf.set_random_seed(self.random_seed)
        self.weights['feature_embedding'] = tf.Variable(
            tf.random_normal([self.feature_num, self.embedding_size], 0.0, 0.01), name='feature_embedding'
        )
        self.weights['feature_bias'] = tf.Variable(
            tf.random_uniform([self.feature_num, 1], 0.0, 0.01), name='feature_bias'
        )

        # deep_layers
        self.input_size = self.field_size * self.embedding_size
        glorot = np.sqrt(2/(self.input_size + self.deep_layers[0]))

        # 输入层
        self.weights['layer_0'] = tf.Variable(
            tf.random_normal([self.input_size, self.deep_layers[0]], 0, glorot), dtype=tf.float32
        )
        self.weights['bias_0'] = tf.Variable(
            tf.random_normal([1, self.deep_layers[0]], 0, glorot), dtype=tf.float32
        )

        # deep层
        for i in range(1, len(self.deep_layers)):
            glorot = np.sqrt(2/(self.deep_layers[i-1] + self.deep_layers[i]))
            self.weights['layer_%d'%(i)] = tf.Variable(
                tf.random_normal([self.deep_layers[i-1], self.deep_layers[i]], 0, glorot), dtype=tf.float32
            )
            self.weights['bias_%d' % (i)] = tf.Variable(
                tf.random_normal([1, self.deep_layers[i]], 0, glorot), dtype=tf.float32
            )

        #final concat layer
        self.input_size = self.deep_layers[-1] +self.embedding_size + self.field_size
        glorot = np.sqrt(2/(self.input_size + 1))
        self.weights['layer_concat'] = tf.Variable(
            tf.random_normal([self.input_size, 1], 0, glorot), dtype=tf.float32
        )
        self.weights['bias_concat'] = tf.Variable(
            tf.constant(0.01), dtype=tf.float32
        )

        return self

    def _init_session(self):
        config = tf.ConfigProto(device_count={"cpu":20},
                                inter_op_parallelism_threads=20,
                                intra_op_parallelism_threads=20,
                                )
        self.saver = tf.train.Saver()
        self.sess = tf.Session(config=config)

        return self

    def adress_data(self, x):
        aa = tf.string_strip(x)
        aa = tf.string_split([aa], 'aaa')
        label, q, d, q_v, d_v = aa.values[0], aa.values[1], aa.values[2], aa.values[3], aa.values[4]
        q = tf.string_split([q], ',').values
        d = tf.string_split([d], ',').values
        q_v = tf.string_split([q_v], ',').values
        d_v = tf.string_split([d_v], ',').values
        q = tf.string_to_number(q, out_type=tf.int32)
        d = tf.string_to_number(d, out_type=tf.int32)
        q_v = tf.string_to_number(q_v, out_type=tf.float32)
        d_v = tf.string_to_number(d_v, out_type=tf.float32)
        label = tf.string_to_number(label, out_type=tf.float32)
        return q, d, q_v, d_v, label

    def batch_norm_layer(self, x, train_phrase, scope_bn):
        bn_train = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=True, reuse=None, trainable=True, scope=scope_bn)
        bn_predict = batch_norm(x, decay=self.batch_norm_decay, center=True, scale=True, updates_collections=None,
                              is_training=False, reuse=True, trainable=True, scope=scope_bn)
        z = tf.cond(train_phrase, lambda: bn_train, lambda: bn_predict)
        return z

    def save_model(self, save_path):
        try:
            print('save model', flush=True)
            self.saver.save(self.sess, save_path)
            print('save done', flush=True)
        except:
            print('save model error', flush=True)

    def load_model(self, load_path):
        try:
            print('load model', flush=True)
            ckpt = tf.train.latest_checkpoint(load_path)
            self.saver.restore(self.sess, ckpt)
            print('load done', flush=True)
        except:
            print('load model error', flush=True)

    def fit(self, file_name, epoch=5):
        q, d, q_v, d_v, train_label = self.load_data(file_name, epoch)
        with tf.variable_scope('DeepFm_model'):
            self._init_graph(q, d, q_v, d_v, train_label)
        self.sess.run(tf.global_variables_initializer())
        try:
            flag = 0
            while True:
                t1 = time()
                epoch_loss, _opt = self.sess.run([self.loss, self.optimizer])
                flag += 1
                print("epoch:%s\tloss:%.6f\ttime:%.6f"%(flag, epoch_loss, time()-t1), flush=True)
        except tf.errors.OutOfRangeError:
            print('end', flush=True)

    def auc(self, y_true, y_pre):
        score = roc_auc_score(y_true, y_pre)
        return score

    def predict(self, q, d, q_v, d_v):
        data_set = tf.data.Dataset.from_tensor_slices((q, d, q_v, d_v))
        data_set = data_set.batch(self.batch_size)
        q, d, q_v, d_v = data_set.make_one_shot_iterator().get_next()
        with tf.variable_scope('DeepFm_model', reuse=True):
            self._init_graph(q, d, q_v, d_v, [0.0], is_train=False)
        y_pre = None
        try:
            print('predict start', flush=True)
            flag = 0
            while True:
                out_batch = self.sess.run([self.out])
                out_batch = out_batch.reshape(1, -1)
                if flag == 0:
                    y_pre = out_batch
                else:
                    y_pre = np.concatenate((y_pre, out_batch), axis=1)
                flag += 1
        except tf.errors.OutOfRangeError:
            print('predict end', flush=True)

        return y_pre[0]

    def transform(self, filename, epoch=1):
        q, d, q_v, d_v, train_label = self.load_data(filename, epoch)
        with tf.variable_scope('DeepFm_model', reuse=True):
            self._init_graph(q, d, q_v, d_v, [0.0], is_train=False)
        y_pre = None
        y_true = None
        try:
            print('transform start', flush=True)
            flag = 0
            while True:
                out_batch, label = self.sess.run([self.out, train_label])
                out_batch = out_batch.reshape(1, -1)
                if flag == 0:
                    y_pre = out_batch
                    y_true = label
                else:
                    y_pre = np.concatenate((y_pre, out_batch), axis=1)
                    y_true = np.concatenate((y_true, label), axis=0)
                flag += 1
        except tf.errors.OutOfRangeError:
            print('transform end', flush=True)

        return y_true, y_pre[0]


if __name__ == '__main__':
    filename = './data/data.txt'
    deep_fm = DeepFm(210000, feature_mean=False, optimizer_type='adam')
    deep_fm.fit(filename, epoch=50)
    deep_fm.save_model('./model/deepfm')
    deep_fm.load_model('./model/')
    y_true , y_pre = deep_fm.transform(filename, epoch=1)
    print(y_true, y_pre)
    score = deep_fm.auc(y_true, y_pre)
    print(score)


