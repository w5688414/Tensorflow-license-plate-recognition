# coding:utf-8
# Author:    xy
# Date:      2021-03-03
# File:      prediction.py
# Description: 预测
import numpy as np
import os
import cv2
import itertools
import sys
sys.path.append("/media/data/projects/license_plate")

from core.util import img_util
import random
import math
from core.util import file_util

from deal_ccpd_data import fetch_plate_img

# E:\\acer\\Documents\\code\\license_plate_recognize-master
from model import license_plate_model


class PredictionModel(object):
    def __init__(self, weight_file):
        self.label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
        self.letters = [letter for letter in self.label_chars]
        self.img_height = 64
        self.img_width = 128
        self.label_max_length = 9
        self.class_num = len([letter for letter in self.label_chars]) + 1
        self.data_format = 'channels_last'
        self.model = self.load_model(weight_file)

    def load_model(self, weight_file):
        model_ = license_plate_model.Model(self.img_height, self.img_width, self.class_num, self.label_max_length)
        model_ = model_.create_model(training=False, data_format=self.data_format)
        model_.load_weights(weight_file)
        return model_

    def decode_label(self, out):
        # 去掉前两行
        out = out[0, 2:]
        score = list(np.max(out, axis=1))
        # 取得每行最大值的索引
        out_best = list(np.argmax(out, axis=1))
        # print(len(out_best))
        # print(score)
        # 分组：相同值归并到一组
        out_best_ = [k for k, g in itertools.groupby(out_best)]
        label = ''
        for index in out_best_:
            if index < len(self.letters):
                label += self.letters[index]
        return label

    def predict_image(self, img):
        """
        预测单张图片
        :param img:
        :return:
        """
        # endswith判断是否以指定后缀结尾
        if img.endswith("jpg") \
                or img.endswith(".JPG") \
                or img.endswith(".png"):
            image = cv2.imread(img, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, (self.img_height, self.img_width))
            if self.data_format == 'channels_last':
                image = image.T
                image = np.expand_dims(image, -1)
                image = np.expand_dims(image, axis=0)
            if self.data_format == 'channels_first':
                image = np.expand_dims(image, -1)
                image = np.expand_dims(image, axis=0)
                image = image.T
            net_out_value = self.model.predict(image)
            predict_str = self.decode_label(net_out_value)
            predict_str = predict_str.strip()
            predict_str = predict_str.strip()
            return predict_str
        return None

    def predict_images(self, img_dir):
        """
        批量预测
        :param img_dir:
        :return:
        """
        output=open('error_vgg.txt','w')
        total = 0
        accuracy = 0
        letter_total = 0
        letter_accuracy = 0
        for root, dirs, files in os.walk(img_dir):
            for img in files:
                predict_str = self.predict_image(os.path.join(root, img))
                predict_str = predict_str.strip()
                plate = img[0:-4]
                plate = plate.upper()
                plate = plate.strip()
                for i in range(min(len(predict_str), len(plate))):
                    if predict_str[i] == plate[i]:
                        letter_accuracy += 1
                letter_total += max(len(predict_str), len(plate))
                success = False
                if predict_str == plate:
                    accuracy += 1
                    success = True
                else:
                    output.write(predict_str+'\t'+plate+'\n')
                total += 1
                print('predict: {0} True: {1}  {2}'.format(predict_str, plate, success))
        print("accuracy : ", accuracy / total)
        print("the accuracy of character: ", letter_accuracy / letter_total)

def video_mirror_output(video):
    new_img = np.zeros_like(video)
    h,w = video.shape[0],video.shape[1]
    for row in range(h):
        for i in range(w):
            new_img[row,i] = video[row,w-i-1]
    return new_img


if __name__ == "__main__":
    current_dir = os.path.dirname(__file__)
    # weight_file = os.path.join('/media/data/projects/license_plate/checkpoints', "14_0.261.h5")
    weight_file = os.path.join('/media/data/projects/license_plate/checkpoints', "13_0.264.h5")
    # weight_file = os.path.join('/media/data/projects/license_plate/checkpoints', "09_0.302.h5")
    # weight_file = os.path.join('/media/data/projects/license_plate/checkpoints', "07_0.390.h5")
    # weight_file = os.path.join(current_dir, "./train_dir/model.hdf5")

    # img_dir = os.path.join(current_dir, "img/pre")
    img_dir = r"/media/data/projects/license_plate/pre1"
    img_file = os.path.join(img_dir)
    # img_file = os.path.join(img_dir, "00A10L30.jpg")
    model = PredictionModel(weight_file)
    # print(model.predict_image(img=img_file))
    # model.predict_images(img_dir=img_dir)

    # cv
    # 创建视频流写入对象，VideoWriter_fourcc为视频编解码器，20为帧播放速率，（640，480）为视频帧大小
    # videoWriter = cv2.VideoWriter('video.avi', cv2.VideoWriter_fourcc('I', '4', '2', '0'), 20, (640, 480))

    img_dir_ = r"/media/data/projects/license_plate/cv"
    save_dir_ = r"/media/data/projects/license_plate/pre1"
    img_file_ = os.path.join(img_dir_, "00A10L30.jpg")

    # cap = cv2.VideoCapture(0)
    #
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # # out = cv2.VideoWriter('output.avi',fourcc,20.0,(640,480))
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (300, 200))
    #
    # while(cap.isOpened()):
    #     ret,frame = cap.read()
    #     frame = video_mirror_output(frame)
    #     if ret==True:
    #         # frame = cv2.flip(frame, 0)
    #         frame = cv2.flip(frame, 180)
    #
    #         # video_mirror_output(frame)
    #         out.write(frame)
    #         cv2.imshow('frame',frame)
    #         if cv2.waitKey(1) & 0xFF == ord('q'):
    #             cv2.imwrite(img_file_, frame)
    #             # cv2.imwrite(img_dir, frame)
    #             fetch_plate_img(img_dir=img_dir_, save_dir=save_dir_)
    #             break
    #     else:
    #         break
    #
    # cap.release()
    # out.release()
    # cv2.destroyAllWindows()

    model.predict_images(img_dir=img_file)
    # model.predict_images(img_dir=img_dir)