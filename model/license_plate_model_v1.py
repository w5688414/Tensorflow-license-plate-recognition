# -*- coding:utf-8 -*-
# Author:      zl
# Date:        2021-03-02
# Description:  CNN + RNN 网络模型
import tensorflow as tf
import os
from tensorflow.keras.layers import *
from tensorflow.keras import Model
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
from tensorflow.keras import layers as Layers

class ResBlock(Model):
    def __init__(self, channels, stride=1):
        super(ResBlock, self).__init__()
        self.flag = (stride != 1)
        self.conv1 = Conv2D(channels, 3, stride, padding='same')
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(channels, 3, padding='same')
        self.bn2 = BatchNormalization()
        self.relu = ReLU()
        if self.flag:
            self.bn3 = BatchNormalization()
            self.conv3 = Conv2D(channels, 1, stride)

    def call(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        if self.flag:
            x = self.conv3(x)
            x = self.bn3(x)
        x1 = Layers.add([x, x1])
        x1 = self.relu(x1)
        return x1



class PlatesModel(object):
    def __init__(self,
                 image_height,
                 image_width,
                 class_num,
                 label_max_length,
                 learning_rate=1e-4):
        self.image_height = image_height
        self.image_width = image_width
        self.channels = 1
        self.class_num = class_num
        self.label_max_length = label_max_length
        self.learning_rate = learning_rate
        self.layer = tf.keras.layers
        self.backend = tf.keras.backend

        self.conv1 = Conv2D(64, 7, 2, padding='same')
        self.bn = BatchNormalization()
        self.relu = ReLU()
        self.mp1 = MaxPooling2D(3, 2)

        self.conv2_1 = ResBlock(64)
        self.conv2_2 = ResBlock(64)
        self.conv2_3 = ResBlock(64)

        self.conv3_1 = ResBlock(128, 2)
        self.conv3_2 = ResBlock(128)
        self.conv3_3 = ResBlock(128)
        self.conv3_4 = ResBlock(128)

        self.conv4_1 = ResBlock(256, 2)
        self.conv4_2 = ResBlock(256)
        self.conv4_3 = ResBlock(256)
        self.conv4_4 = ResBlock(256)
        self.conv4_5 = ResBlock(256)
        self.conv4_6 = ResBlock(256)

        self.conv5_1 = ResBlock(512, 2)
        self.conv5_2 = ResBlock(512)
        self.conv5_3 = ResBlock(512)

        self.pool = GlobalAveragePooling2D()
        self.fc1 = Dense(512, activation='relu')
        self.dp1 = Dropout(0.5)
        self.fc2 = Dense(512, activation='relu')
        self.dp2 = Dropout(0.5)
        self.fc3 = Dense(64)
        
        

    def build_resnet(self, inputs, data_format):
        """
        vgg16
        :param inputs: self.layer.Input(shape=(64, 128, 1), dtype=tf.float32)
        :param data_format: channel_last
        :return:
        """
        """
        layer1 (64, 128, 64)
        """
        x = self.conv1(inputs)
        x = self.bn(x)
        x = self.relu(x)
        x = self.mp1(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)

        x = self.conv3_1(x)
        x = self.conv3_2(x)
        x = self.conv3_3(x)
        x = self.conv3_4(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.conv4_3(x)
        x = self.conv4_4(x)
        x = self.conv4_5(x)
        x = self.conv4_6(x)

        x = self.conv5_1(x)
        x = self.conv5_2(x)
        # print(x)
        x = self.conv5_3(x)
        # print(x)
        x = Reshape(target_shape=(32, 128), name='reshape')(x)
        resnet = Dense(64, activation=tf.nn.relu, name='dense1')(x)
        # x = self.pool(x)
        # x = self.fc1(x)
        # x = self.dp1(x)
        # x = self.fc2(x)
        # x = self.dp2(x)
        # resnet = self.fc3(x)
        return resnet

    def build_rnn(self, tensor):
        """
        双向LSTM
        :param tensor: shape=(32, 64)
        :return:
        """
        tensor = self.layer.Bidirectional(self.layer.LSTM(128, return_sequences=True), merge_mode='concat', name="bidirectional_GRU")(tensor)
        tensor = self.layer.Dense(self.class_num, name='dense2')(tensor)
        return tensor

    def ctc_lambda_func(self, args):
        y_pre, labels, input_len, label_len = args
        y_pre = y_pre[:, 2:, :]
        return self.backend.ctc_batch_cost(labels, y_pre, input_len, label_len)

    def create_model(self, data_format, training=True):
        if data_format == 'channels_first':
            # (batch, channels, height, width) default
            input_shape = (self.channels, self.image_height, self.image_width)
        else:
            # (batch, height, width, channels)
            assert data_format == 'channels_last'
            input_shape = (self.image_height, self.image_width, self.channels)
        inputs = self.layer.Input(shape=input_shape, name='input', dtype=tf.float32)
        # cnn
        tensor = self.build_resnet(inputs=inputs, data_format=data_format)
        # rnn
        # 在采用CNN提取图像卷积特征后，然后RNN进一步提取图像卷积特征中的序列特征。
        tensor = self.build_rnn(tensor)
        # ctc
        y_pre = self.layer.Activation('softmax', name='softmax')(tensor)
        labels = self.layer.Input(name='labels', shape=[self.label_max_length], dtype=tf.float32)
        input_len = self.layer.Input(name='input_length', shape=[1], dtype=tf.int64)
        label_len = self.layer.Input(name='label_length', shape=[1], dtype=tf.int64)
        loss_out = self.layer.Lambda(self.ctc_lambda_func, output_shape=(1,), name='ctc')(
            [y_pre, labels, input_len, label_len])
        if training:
            return tf.keras.models.Model(inputs=[inputs, labels, input_len, label_len], outputs=loss_out)
        else:
            return tf.keras.models.Model(inputs=[inputs], outputs=y_pre)

