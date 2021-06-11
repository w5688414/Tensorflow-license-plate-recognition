# coding:utf-8
# Description:  训练
import sys
sys.path.insert(0,"/media/data/projects/license_plate")
from model.resnet_gru import PlatesModel
from model.data_generator import ImageGenerator
import tensorflow as tf
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau


def ctc_loss_func():
    return {'ctc': lambda y_true, y_pred: y_pred}


def train(train_dir, train_img_dir, eval_img_dir, hdf5=None):
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.isdir(train_img_dir):
        print("The folder does not exist: {0}".format(train_img_dir))
    if not os.path.isdir(eval_img_dir):
        print("The folder does not exist: {0}".format(eval_img_dir))
    label_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
    img_height = 64
    img_width = 128
    label_max_length = 9
    # data_format = 'channels_first'  # BCHW
    data_format = 'channels_last'     # BHWC
    train_generator = ImageGenerator(data_format=data_format, label_chars=label_chars, img_dir=train_img_dir,
                                     img_height=img_height, img_width=img_width, batch_size=128,
                                     label_max_length=label_max_length)
    eval_generator = ImageGenerator(data_format=data_format, label_chars=label_chars, img_dir=eval_img_dir,
                                    img_height=img_height, img_width=img_width, batch_size=16,
                                    label_max_length=label_max_length)
    checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=train_dir + '/{epoch:02d}_{val_loss:.3f}.h5',
                                                    monitor='val_loss',save_best_only=True, save_weights_only=True,
                                                    verbose=1, period=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2,
                              patience=5, min_lr=0.001)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = PlatesModel(img_height,
                  img_width,
                  train_generator.class_num,
                  label_max_length).create_model(data_format=data_format)


    #
    # model.compile(optimizer=tf.keras.optimizers.Adadelta(), loss=ctc_loss_func())
    # model.compile(optimizer=tf.keras.optimizers.A, loss=ctc_loss_func())
    # model.compile(optimizer=tf.keras.optimizers.Adagrad(), loss=ctc_loss_func())
    # model.load_weights('../checkpoints/09_0.405.h5')

    model.compile(optimizer=tf.keras.optimizers.Adam(), loss=ctc_loss_func())
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.steps_per_epoch,
                        epochs=20,
                        callbacks=[checkpoint,reduce_lr],
                        validation_data=eval_generator,
                        use_multiprocessing=False,
                        validation_steps=eval_generator.steps_per_epoch)


if __name__ == '__main__':
    train_dir = r"/media/data/projects/license_plate/checkpoints_resnet_gru"
    train_img_dir = r"/media/data/projects/license_plate/ccpd_new/train/"
    eval_img_dir = r"/media/data/projects/license_plate/ccpd_new/eval/"
    train(train_dir, train_img_dir, eval_img_dir)

# "F:\coding\license_plate\ccpd_base"
# "/home/zl/code/plate_recognition/plate_train"
# "/home/zl/code/plate_recognition/model/train_dir1"
# "/home/zl/code/plate_recognition/plate_eval"




