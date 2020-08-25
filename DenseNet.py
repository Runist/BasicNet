# -*- coding: utf-8 -*-
# @File : DenseNet.py
# @Author: Runist
# @Time : 2020/8/24 9:17
# @Software: PyCharm
# @Brief: 用keras实现DenseNet

import tensorflow as tf
import os
from tensorflow.keras import layers, losses, optimizers, models, callbacks, applications
import numpy as np


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


def conv_block(x, growth_rate, dropout_rate=None):
    """
    DenseNet的conv块，以论文的描述是BN-ReLU-Conv
    :param x: 输入变量
    :param growth_rate: 增长率
    :param dropout_rate: dropout的比率
    :return: x
    """
    x1 = layers.BatchNormalization()(x)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(4 * growth_rate, kernel_size=1, use_bias=False)(x1)
    if dropout_rate:
        x1 = layers.Dropout(dropout_rate)(x1)

    x1 = layers.BatchNormalization()(x1)
    x1 = layers.Activation('relu')(x1)
    x1 = layers.Conv2D(growth_rate, padding='same', kernel_size=3, use_bias=False)(x1)
    if dropout_rate:
        x1 = layers.Dropout(dropout_rate)(x1)

    x = layers.Concatenate()([x, x1])

    return x


def transition_block(x, reduction):
    """
    过渡层，每个Dense Block直接降采样的部分
    :param x: 输入
    :param reduction: 维度降低的部分
    :return: x
    """
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    # 降维
    x = layers.Conv2D(int(x.shape[-1] * reduction), kernel_size=1, use_bias=False)(x)
    x = layers.AveragePooling2D(2, strides=2)(x)

    return x


def dense_block(x, blocks, growth_rate, dropout_rate=None):
    """
    一个dense block由多个卷积块组成
    :param x: 输入
    :param blocks: 每个dense block卷积多少次
    :param dropout_rate: dropout的比率
    :param growth_rate: 每个特征层的增长率
    :return: x
    """
    for _ in range(blocks):
        x = conv_block(x, growth_rate, dropout_rate)
    return x


def DenseNet(height, width, channel, blocks, class_num, growth_rate=32, reduction=0.5, dropout_rate=None):
    """
    建立DenseNet网络，需要调节dense block的数量、一个dense block中有多少个conv、growth_rate、reduction、dropout rate
    :param height: 图像的高
    :param width: 图像的宽
    :param channel: 图像的通道数
    :param blocks: 卷积块的数量
    :param class_num: 分类的数量
    :param growth_rate: 每个特征层的增长率
    :param reduction: 过渡层减少层数的比例
    :param dropout_rate: dropout的比率
    :return: model
    """
    input_image = layers.Input((height, width, channel), dtype="float32")

    x = layers.ZeroPadding2D(padding=((3, 3), (3, 3)))(input_image)
    x = layers.Conv2D(64, 7, strides=2, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)))(x)
    x = layers.MaxPooling2D(3, strides=2)(x)

    for block in blocks[:-1]:
        x = dense_block(x, block, growth_rate=growth_rate, dropout_rate=dropout_rate)
        x = transition_block(x, reduction=reduction)
    x = dense_block(x, blocks[-1], growth_rate=growth_rate, dropout_rate=dropout_rate)

    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(class_num, activation='softmax')(x)

    model = models.Model(input_image, x)
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
           callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

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
    img_path = './dataset/tulips.jpg'

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
    weights_path = "./logs/weights/DenseNet.h5"
    width = height = 224
    channel = 3

    batch_size = 32
    num_classes = 5
    epochs = 20
    lr = 0.0003
    growth_rate = 12
    reduction = 0.5
    is_train = False
    if is_train:
        dropout_rate = 0.2
    else:
        dropout_rate = None

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
    model = DenseNet(height, width, channel, [8, 16, 8], num_classes,
                     growth_rate=growth_rate,
                     reduction=reduction,
                     dropout_rate=dropout_rate)

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

