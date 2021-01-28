# -*- coding: utf-8 -*-
# @File : VGG.py
# @Author: Runist
# @Time : 2020/3/2 12:31
# @Software: PyCharm
# @Brief: VGG网络实现


import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, models, callbacks
import os
import numpy as np


cfgs = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']
}


def read_data(path):
    """
    读取数据，传回图片完整路径列表 和 仅有数字索引列表
    :param path: 数据集路径
    :return: 图片路径列表、数字索引列表
    """
    image_list = list()
    label_list = list()
    class_list = os.listdir(path)

    for i, value in enumerate(class_list):
        dirs = os.path.join(path, value)
        for pic in os.listdir(dirs):
            pic_full_path = os.path.join(dirs, pic)
            image_list.append(pic_full_path)
            label_list.append(i)

    return image_list, label_list


def make_datasets(image, label, batch_size, mode):
    """
    将图片和标签合成一个 数据集
    :param image: 图片路径
    :param label: 标签路径
    :param batch_size: 批处理的数量
    :param mode: 处理不同数据集的模式
    :return: dataset
    """
    # 这是GPU读取方式
    dataset = tf.data.Dataset.from_tensor_slices((image, label))
    if mode == 'train':
        # 打乱数据，这里的shuffle的值越接近整个数据集的大小，越贴近概率分布。但是电脑往往没有这么大的内存，所以适量就好
        dataset = dataset.shuffle(buffer_size=len(label))
        # map的作用就是根据定义的 函数，对整个数据集都进行这样的操作
        # 而不用自己写一个for循环，如：可以自己定义一个归一化操作，然后用.map方法都归一化
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        # prefetch解耦了 数据产生的时间 和 数据消耗的时间
        # prefetch官方的说法是可以在gpu训练模型的同时提前预处理下一批数据
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)

    return dataset


def parse(img_path, label, width=224, height=224, class_num=5):
    """
    对数据集批量处理的函数
    :param img_path: 必须有的参数，图片路径
    :param label: 必须有的参数，图片标签（都是和dataset的格式对应）
    :param class_num: 类别数量
    :param height: 图像高度
    :param width: 图像宽度
    :return: 单个图片和分类
    """
    label = tf.one_hot(label, depth=class_num)
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [width, height])

    return image, label


def features(cfg):
    """
    由于VGG有不同层数的配置，根据传入的不同种结构的字典，构建vgg网络
    :param cfg: 根据传入的列表数据构建vgg卷积层
    :return: Sequential 对象
    """

    feature_layers = []
    for v in cfg:
        if v == 'M':
            feature_layers.append(layers.MaxPool2D(pool_size=2, strides=2))
        else:
            conv2d = layers.Conv2D(v, kernel_size=3, padding='SAME', activation='relu')
            feature_layers.append(conv2d)

    return models.Sequential(feature_layers, name='feature')


def VGG(feature, height, width, channel, class_num=5):
    """
    建立VGG网络，由于VGG后面的全连接层都是一样的，所以封装成函数
    :param feature: 前十几层不同，放在一个函数中搭建
    :param width: 宽度
    :param height: 高度
    :param channel: 图片通道数
    :param class_num: 分类数量
    :return: model
    """
    # Tensor的通道排序是NHWC
    input_image = layers.Input(shape=(height, width, channel), dtype="float32")
    x = feature(input_image)
    x = layers.Flatten()(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dropout(rate=0.5)(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(class_num)(x)
    output = layers.Softmax()(x)
    model = models.Model(inputs=input_image, outputs=output)
    model.summary()

    return model


def model_train(model, x_train, x_val, epochs, train_step, val_step, weights_path):
    """
    模型训练
    :param model: 定义好的模型
    :param x_train: 训练集数据
    :param x_val: 验证集数据
    :param epochs: 迭代次数
    :param train_step: 一个epoch的训练次数
    :param val_step: 一个epoch的验证次数
    :param weights_path: 权值保存路径
    :return: None
    """
    # 如果选成h5格式，则不会保存成ckpt的tensorflow常用格式
    # monitor是指验证参数，如何评估模型好坏的标准
    cbk = [callbacks.ModelCheckpoint(filepath=weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
           callbacks.EarlyStopping(patience=10, min_delta=1e-3)]

    history = model.fit(x_train,
                        steps_per_epoch=train_step,
                        epochs=epochs,
                        validation_data=x_val,
                        validation_steps=val_step,
                        callbacks=cbk,
                        verbose=1)

    # 如果只希望在结束训练后保存模型，则可以直接调用save_weights和save，这二者的区别就是一个只保存权值文件，另一个保存了模型结构
    # model.save_weights(weights_path)


def model_predict(model, weights_path, height, width):
    """
    模型预测
    :param model: 定义好的模型，因为保存的时候只保存了权重信息，所以读取的时候只读取权重，则需要网络结构
    :param weights_path: 权重文件的路径
    :param height: 图像高度
    :param width: 图像宽度
    :return: None
    """
    class_indict = ['daisy', 'dandelion', 'rose', 'sunflower', 'tulips']
    img_path = './dataset/sunflower.jpg'

    # 值得一提的是，这里开启图片如果用其他方式，需要考虑读入图片的通道数，在制作训练集时采用的是RGB，而opencv采用的则是BGR
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])

    # 输入到网络必须是一个batch(batch_size, height, weight, channels)
    # 用这个方法去扩充一个维度
    image = (np.expand_dims(image, 0))

    model.load_weights(weights_path)
    # 预测的结果是包含batch这个维度，所以要把这个batch这维度给压缩掉
    result = np.squeeze(model.predict(image))
    predict_class = int(np.argmax(result))
    print("预测类别：{}, 预测可能性{:.03f}".format(class_indict[predict_class], result[predict_class]*100))


def main():
    dataset_path = './dataset/'
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    weights_path = "./model_weights/VGG.h5"
    width = height = 224
    channel = 3

    batch_size = 32
    num_classes = 5
    epochs = 20
    lr = 0.0003
    is_train = False

    # 选择编号为0的GPU，如果不使用gpu则置为-1
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 这里的操作是让GPU动态分配内存不要将GPU的所有内存占满
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    # 数据读取
    train_image, train_label = read_data(train_dir)
    val_image, val_label = read_data(val_dir)

    train_step = len(train_label) // batch_size
    val_step = len(val_label) // batch_size

    train_dataset = make_datasets(train_image, train_label, batch_size, mode='train')
    val_dataset = make_datasets(val_image, val_label, batch_size, mode='validation')

    # 模型搭建
    vgg13_feature = features(cfgs['vgg13'])
    model = VGG(vgg13_feature, height, width, channel, num_classes)

    model.compile(loss=losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])
    if is_train:
        # 模型训练
        model_train(model, train_dataset, val_dataset, epochs, train_step, val_step, weights_path)
    else:
        # 模型预测
        model_predict(model, weights_path, height, width)


if __name__ == '__main__':
    main()
