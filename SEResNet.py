# -*- coding: utf-8 -*-
# @File : SEResNet.py
# @Author: Runist
# @Time : 2020-12-30 10:05
# @Software: PyCharm
# @Brief:

from tensorflow.keras import layers, models, optimizers, losses, callbacks
import numpy as np
import tensorflow as tf
import os


class SEBasicBlock(layers.Layer):

    def __init__(self, out_channel, strides=1, downsample=False, reduce_ratio=2, use_se_block=True, **kwargs):
        """

        :param out_channel: 输出通道
        :param strides: 卷积步长
        :param downsample: 是否进行下采样
        :param kwargs: 变长层名字
        """
        super(SEBasicBlock, self).__init__(**kwargs)

        self.downsample = downsample
        if downsample:
            self.downsample_conv = layers.Conv2D(out_channel, kernel_size=1, strides=strides, use_bias=False)
            self.downsample_bn = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(out_channel, kernel_size=3, strides=strides, padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

        self.se = use_se_block
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.reshape = layers.Reshape((1, 1, out_channel))
        self.fc_1 = layers.Dense(out_channel // reduce_ratio, activation='relu')
        self.fc_2 = layers.Dense(out_channel, activation='sigmoid')
        self.scale = layers.Multiply()

    def call(self, inputs, training=False):
        """
        对象调用函数
        :param inputs: block输入
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :return:
        """
        identity = inputs

        if self.downsample:
            identity = self.downsample_conv(identity)
            identity = self.downsample_bn(identity)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)

        if self.se:
            se_input = x
            x = self.avg_pool(x)
            x = self.reshape(x)
            x = self.fc_1(x)
            x = self.fc_2(x)
            x = self.scale([x, se_input])

        x = self.add([identity, x])
        x = self.relu(x)

        return x


class SEBottleneckBlock(layers.Layer):

    def __init__(self, out_channel, strides=1, downsample=False, reduce_ratio=2, use_se_block=False, **kwargs):
        """

        :param out_channel: 输出通道
        :param strides: 卷积步长
        :param downsample: 是否进行下采样
        :param kwargs: 变长层名字
        """
        super(SEBottleneckBlock, self).__init__(**kwargs)

        self.downsample = downsample

        self.shortcut_conv = layers.Conv2D(out_channel*4, kernel_size=1, strides=strides, use_bias=False)
        self.shortcut_bn = layers.BatchNormalization()

        self.conv1 = layers.Conv2D(out_channel, kernel_size=1, strides=strides, padding="SAME", use_bias=False)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(out_channel, kernel_size=3, strides=1, padding="SAME", use_bias=False)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(out_channel*4, kernel_size=1, strides=1, padding="SAME", use_bias=False)
        self.bn3 = layers.BatchNormalization()

        self.relu = layers.ReLU()
        self.add = layers.Add()

        self.se = use_se_block
        self.avg_pool = layers.GlobalAveragePooling2D()
        self.reshape = layers.Reshape((1, 1, out_channel))
        self.fc_1 = layers.Dense(out_channel // reduce_ratio, activation='relu')
        self.fc_2 = layers.Dense(out_channel, activation='sigmoid')
        self.scale = layers.Multiply()

    def call(self, inputs, training=False):
        """
        对象调用函数
        :param inputs: block输入
        :param training: 用在训练过程和预测过程中，控制其生效与否
        :return:
        """
        identity = inputs
        if self.downsample:
            x = self.shortcut_conv(inputs)
            x = self.shortcut_bn(x)
            identity = self.relu(x)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)

        if self.se:
            se_input = x
            x = self.avg_pool(x)
            x = self.reshape(x)
            x = self.fc_1(x)
            x = self.fc_2(x)
            x = self.scale([x, se_input])

        x = self.add([identity, x])
        x = self.relu(x)

        return x


def resblock_body(res_block, filters, num_blocks, strides, name):
    """
    ResNet中残差单元
    :param res_block: 残差块类型
    :param filters: 卷积核个数
    :param num_blocks: 残差块重复的次数
    :param strides: 步长
    :param name: 该残差单元的名字
    :return:
    """
    if res_block == SEBasicBlock:
        layer_list = [res_block(filters, downsample=strides != 1, strides=strides)]
    else:
        layer_list = [res_block(filters, downsample=True, strides=strides)]

    for _ in range(num_blocks - 1):
        layer_list.append(res_block(filters))

    return models.Sequential(layer_list, name=name)


def SE_ResNet(height, width, num_class, res_block, blocks_list):
    """
    ResNet网络结构，通过传入不同的残差块和重复的次数进行不同层数的ResNet构建
    :param height: 网络输入宽度
    :param width: 网络输入高度
    :param num_class: 分类数量
    :param res_block: 残差块单元
    :param blocks_list: 每个残差单元重复的次数列表
    :return:
    """
    input_image = layers.Input(shape=(height, width, 3))
    x = layers.Conv2D(filters=64, kernel_size=7, strides=2, padding='SAME', name='conv1', use_bias=False)(input_image)
    x = layers.BatchNormalization(name="conv1/BatchNorm")(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name="max_pool")(x)

    x = resblock_body(res_block, 64, blocks_list[0], strides=1, name='conv2_x')(x)
    x = resblock_body(res_block, 128, blocks_list[1], strides=2, name='conv3_x')(x)
    x = resblock_body(res_block, 256, blocks_list[2], strides=2, name='conv4_x')(x)
    x = resblock_body(res_block, 512, blocks_list[3], strides=2, name='conv5_x')(x)

    x = layers.GlobalAvgPool2D(name='avg_pool')(x)
    x = layers.Dense(num_class, name="logits")(x)
    outputs = layers.Softmax()(x)

    model = models.Model(inputs=input_image, outputs=outputs)
    model.summary()

    return model


def SE_ResNet18(height, width, num_class):
    return SE_ResNet(height, width, num_class, SEBasicBlock, [2, 2, 2, 2])


def SE_ResNet34(height, width, num_class):
    return SE_ResNet(height, width, num_class, SEBasicBlock, [3, 4, 6, 3])


def SE_ResNet50(height, width, num_class):
    return SE_ResNet(height, width, num_class, SEBottleneckBlock, [3, 4, 6, 3])


def SE_ResNet101(height, width, num_class):
    return SE_ResNet(height, width, num_class, SEBottleneckBlock, [3, 4, 23, 3])


def read_data(path, class_list):
    """
    读取数据，传回图片完整路径列表 和 仅有数字索引列表
    :param path: 数据集路径
    :param class_list: 标签列表，防止因操作系统不同os读取顺序不同
    :return: 图片路径列表、数字索引列表
    """
    image = []
    label = []

    for i, value in enumerate(class_list):
        dirs = os.path.join(path, value)
        for pic in os.listdir(dirs):
            pic_full_path = os.path.join(dirs, pic)
            image.append(pic_full_path)
            label.append(i)

    return image, label


def train_parse(img_path, label, class_num=5):
    """
    对数据集批量处理的函数
    :param img_path: 图片路径
    :param class_num: 类别数量
    :param label: 图片标签对应数字所引
    :return: 单个图片和分类
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [224, 224])

    # 在训练集中，有50%的概率增强数据
    if np.random.random() < 0.5:
        if np.random.random() < 0.5:
            image = tf.image.random_flip_left_right(image)
        if np.random.random() < 0.5:
            image = tf.image.random_flip_up_down(image)
        if np.random.random() < 0.5:
            image = tf.image.random_brightness(image, 0.2)
        if np.random.random() < 0.5:
            image = tf.image.random_contrast(image, 0.3, 2.0)
        if np.random.random() < 0.5:
            image = tf.image.random_hue(image, 0.15)
        if np.random.random() < 0.5:
            image = tf.image.random_saturation(image, 0.3, 2.0)

    image /= 255.

    label = tf.one_hot(label, depth=class_num)
    return image, label


def val_parse(img_path, label, class_num=5):
    """
    对数据集批量处理的函数
    :param img_path: 图片路径
    :param class_num: 类别数量
    :param label: 图片标签对应数字所引
    :return: 单个图片和分类
    """
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.resize(image, [224, 224])
    image /= 255.

    label = tf.one_hot(label, depth=class_num)
    return image, label


def make_datasets(image, label, batch_size, mode):
    """
    将图片和标签合成一个 数据集
    :param image: 图片路径
    :param label: 标签路径
    :param batch_size: 批处理的数量
    :param mode: 处理不同数据集的模式
    :return: dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((image, label))
    if mode == 'train':
        dataset = dataset.shuffle(buffer_size=len(label))
        dataset = dataset.map(train_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat()
        dataset = dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    else:
        dataset = dataset.map(val_parse, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)

    return dataset


def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer._decayed_lr("float32")
    return lr


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)

    train_dir = './dataset/train'
    val_dir = './dataset/validation'
    epochs = 200
    batch_size = 256
    lr = 2e-3
    class_name = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
    num_classes = len(class_name)
    is_train = False

    train_image, train_label = read_data(train_dir, class_name)
    val_image, val_label = read_data(val_dir, class_name)
    train_step = len(train_image) // batch_size
    val_step = len(val_image) // batch_size

    train_dataset = make_datasets(train_image, train_label, batch_size, mode='train')
    val_dataset = make_datasets(val_image, val_label, batch_size, mode='train')

    model = SE_ResNet18(224, 224, num_classes)

    optimizer = optimizers.Adam(lr)
    lr_metric = get_lr_metric(optimizer)
    model.compile(optimizer=optimizer,
                  loss=losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy', lr_metric])

    cbk = [callbacks.ModelCheckpoint("./model_weights/SEResNet.h5", save_weights_only=True, save_best_only=True),
           callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2)]

    if is_train:
        model.fit(train_dataset,
                  steps_per_epoch=train_step,
                  epochs=epochs,
                  validation_data=val_dataset,
                  validation_steps=val_step,
                  callbacks=cbk,
                  verbose=1)
    else:
        model.load_weights("./model_weights/SEResNet.h5")
        img_path = './dataset/dandelion.jpg'
        image, _ = val_parse(img_path, 0)
        pred = model.predict(tf.expand_dims(image, axis=0))[0]

        index = tf.argmax(pred).numpy()
        print("预测类别：{}, 预测可能性{:.03f}".format(class_name[index], pred[index]*100))


if __name__ == '__main__':
    main()

