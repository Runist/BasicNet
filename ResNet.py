# -*- coding: utf-8 -*-
# @File : ResNet.py
# @Author: Runist
# @Time : 2020/3/4 12:21
# @Software: PyCharm
# @Brief: 残差网络的使用

import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, models, callbacks
import os
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


class BasicBlock(layers.Layer):
    # 用来控制不同层次的残差网络的通道倍增数
    expansion = 1

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        """

        :param out_channel:
        :param strides:
        :param downsample: 对应下采样的卷积处理方法
        :param kwargs: 变长层名字
        """
        super(BasicBlock, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides, padding="SAME", use_bias=False)
        # epsilon是防止分母为0的情况
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        """

        :param inputs:
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :return:
        """
        # identity代表的是残差网络基本单元的输入，最后要和输出相加
        identity = inputs
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class Bottleneck(layers.Layer):
    # 用来控制不同层次的残差网络的通道倍增数
    expansion = 4

    def __init__(self, out_channel, strides=1, downsample=None, **kwargs):
        """

        :param out_channel:
        :param strides:
        :param downsample: 对应下采样的卷积处理方法
        :param kwargs: 变长层名字
        """
        super(Bottleneck, self).__init__(**kwargs)
        self.conv1 = layers.Conv2D(out_channel,
                                   kernel_size=1, strides=1, padding="SAME", use_bias=False, name='conv1')
        # epsilon是防止分母为0的情况
        self.bn1 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv1/BatchNorm')
        # -------------------------------------------------
        self.conv2 = layers.Conv2D(out_channel,
                                   kernel_size=3, strides=strides, padding="SAME", use_bias=False, name='conv2')
        self.bn2 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv2/BatchNorm')
        # -------------------------------------------------
        self.conv3 = layers.Conv2D(out_channel*self.expansion,
                                   kernel_size=1, strides=1, padding="SAME", use_bias=False, name='conv3')
        self.bn3 = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name='conv3/BatchNorm')
        # -------------------------------------------------
        self.downsample = downsample
        self.relu = layers.ReLU()
        self.add = layers.Add()

    def call(self, inputs, training=False):
        """

        :param inputs:
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :return:
        """
        # identity代表的是残差网络基本单元的输入，最后要和输出相加
        identity = inputs

        # 如果有downsample就代表使用论文中带有维度变换的shortcut
        if self.downsample is not None:
            identity = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        x = self.add([identity, x])
        x = self.relu(x)

        return x


def make_layer(block, in_channel, first_channel, block_list, name, strides=1):
    """
    构建一个大的layer，里面包含几个残差单元
    :param block: 使用的 哪种残差单元 对应的类方法
    :param in_channel: 上一层的通道数
    :param first_channel: 对应这一块残差单元的第一层通道数
    :param block_list: 对应 有几个残差单元
    :param name: 这个 大块的类残差单元的名字
    :param strides: 卷积的步长
    :return:
    """
    downsample = None

    # 一般18、34层第一个残差块的shortcut是没有虚线结构的，所以不需要传入downsample方法
    # 所以步长就为2的时候就可以进入if语句
    # 而50层以上的第一个残差块也是要降采样，50层以上和18、34有所不同的是，他们的残差块输出层通道数是输入层的4倍
    # 当然可以是其他倍数、只不过这个是经过作者测试比较稳定，而浅层的输入输出都是一样的 的通道数
    # 所以利用 上一层的通道数 是否等于 残差块第一层 * 类中倍增系数 就可以让 更深层次的情况包括进去
    if strides != 1 or in_channel != first_channel * block.expansion:
        # 在这里定义block内的downsample方法
        # 然后作为Block类 的参数传入
        downsample = models.Sequential([
            # 无论是18、34层的还是更深层次的，其输出的通道数都是以first_channel*block.expansion为结果
            # 如果 是浅层的结构就是1*first_channel, 若是深层就是4*first_channel
            layers.Conv2D(first_channel*block.expansion, kernel_size=1, strides=strides, use_bias=False, name="conv1"),
            layers.BatchNormalization(momentum=0.9, epsilon=1.001e-5, name="BatchNorm")
        ], name="shortcut")

    # 无论是浅层还是深层，只有第一层是虚线残差结构，后面的层次都是实线残差结构
    layer_list = [block(first_channel, downsample=downsample, strides=strides, name="unit_1")]

    for i in range(1, block_list):
        layer_list.append(block(first_channel, name="unit_{}".format(i+1)))

    return models.Sequential(layer_list, name=name)


def ResNet(block, blocks_list, height, width, num_class, include_top=False):
    """
    构建残差网络的框架，我们把残差网络的基本单元称为残差单元，
    残差单元内部 又由标准的 残差结构组成：包括2层或3层卷积层、批归一化、激活函数
    而残差网络都是由conv1 + conv2_x + conv3_x + conv4_x + conv5_x构成
    不同的层数的残差网络，对应的conv内的结构也有所不同（18、34是用BasicBlock残差单元,
    50以上的用的是Bottleneck残差单元）。
    :param block: 使用的残差单元定义
    :param blocks_list: 每个conv中对应残差单元的个数 列表
    :param height:
    :param width:
    :param num_class:
    :param include_top: 是否添加全连接层，并激活顶层函数
    :return: model
    """
    # input_shape(None, 224, 224, 3)
    input_image = layers.Input(shape=(height, width, 3), dtype='float32')
    # 每个残差网络第一层都是7x7的卷积+3x3池化
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='SAME', name='conv1', use_bias=False)(input_image)
    x = layers.BatchNormalization(momentum=0.9, epsilon=1e-5, name="conv1/BatchNorm")(x)        # (112, 112, 64)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME')(x)     # (56, 56, 64)

    x = make_layer(block, x.shape[-1], 64, blocks_list[0], name="conv2_x")(x)     # (56, 56, 64)
    x = make_layer(block, x.shape[-1], 128, blocks_list[1], strides=2, name="conv3_x")(x)       # (28, 28, 128)
    x = make_layer(block, x.shape[-1], 256, blocks_list[2], strides=2, name="conv4_x")(x)       # (14, 14, 256)
    x = make_layer(block, x.shape[-1], 512, blocks_list[3], strides=2, name="conv5_x")(x)       # (7, 7, 256)

    if include_top:
        # 使用了全局平均池化，无论输入矩阵高和宽多少，都换变成1x1且 平铺处理
        x = layers.GlobalAvgPool2D()(x)  # pool + flatten
        x = layers.Dense(num_class, name="logits")(x)
        predict = layers.Softmax()(x)
    else:
        # 如果不展平处理激活的话，就可以在后面使用迁移学习做更多的操作
        predict = x

    model = models.Model(inputs=input_image, outputs=predict)
    model.summary()

    return model


def ResNet18(height, width, num_class):
    return ResNet(BasicBlock, [2, 2, 2, 2], height, width, num_class, include_top=True)


def ResNet34(height, width, num_class):
    return ResNet(BasicBlock, [3, 4, 6, 4], height, width, num_class, include_top=True)


def ResNet50(height, width, num_class):
    return ResNet(Bottleneck, [3, 4, 6, 3], height, width, num_class, include_top=True)


def ResNet101(height, width, num_class):
    return ResNet(Bottleneck, [3, 4, 23, 3], height, width, num_class, include_top=True)


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
    cbk = [callbacks.ModelCheckpoint(filepath=weights_path,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     monitor='val_loss'),
           callbacks.EarlyStopping(patience=5, min_delta=1e-3)]

    # 重点：fit 和 fit_generator的区别
    # 之前fit方法是使用整个训练集可以放入内存当中
    # fit_generator的就是用在应用于数据集非常大的时候，但2.1已经整合在fit里面了现在已经改了。
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
    weights_path = "./logs/weights/ResNet.h5"

    width = height = 224
    batch_size = 32
    num_classes = 5
    epochs = 30
    lr = 0.0003
    is_train = False

    # 选择编号为0的GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    # 这里的操作是让GPU动态分配内存不要将GPU的所有内存占满，多人协同时合理分配CPU
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

    # 定义模型
    model = ResNet18(width, height, num_classes)

    # 输出层如果已经经过softmax激活就用from_logits置为False，如果没有处理 就置为True
    # 如果没有处理，模型会更加稳定
    model.compile(loss=losses.CategoricalCrossentropy(from_logits=False),
                  optimizer=optimizers.Adam(learning_rate=lr),
                  metrics=["accuracy"])

    if is_train:
        # 模型训练
        model_train(model, train_dataset, val_dataset, epochs, train_step, val_step, weights_path)
    else:
        # 模型预测
        model_predict(model, weights_path, height, width)


if __name__ == "__main__":
    main()
