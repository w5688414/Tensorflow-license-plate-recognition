# Tensorflow-license-plate-recognition

车牌识别项目

## 训练经验

+ VGG+LSTM+RMSprop能够达到99.5%的准确率
+ VGG+LSTM+Adam不超过95%的准确率，这也是我第一次发现Adam过早的收敛到局部最优哈

## 操作步骤

训练和测试的文件在models里面哈


1. 神经网络：license_plate_model.py 这是vggbackbone，还有license_plate_model_v1.py 这是resnet的backbone
2. 训练+评估：train.py
3. 数据生成器：data_generator.py
4. 预测/测试：prediction.py

* 训练模型为model/train_dir/model.hdf5，该模型训练了13个epoch，能够达到准去率为99.5%


## 使用训练的模型进行测试

配置好环境好，运行python prediction.py即可进行训练好的model.hdf5进行测试