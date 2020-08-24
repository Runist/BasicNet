# -*- coding: utf-8 -*-
# @File : exam5_GoogLeNet.py
# @Author: Runist
# @Time : 2020/3/2 15:41
# @Software: PyCharm
# @Brief: GoogLeNet的使用
import tensorflow as tf
import os
from tensorflow.keras import layers, losses, optimizers, models, callbacks, metrics
import numpy as np
from tqdm import tqdm


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


class Inception(layers.Layer):
    """
    Inception结构
    经过此卷积之后，最后输出的通道层数为：conv2d1x1 + conv2d3x3 + conv2d5x5 + pool_pro
    """
    def __init__(self, conv2d1x1, conv2d3x3red, conv2d3x3, conv2d5x5red, conv2d5x5, pool_pro, **kwargs):
        """
        Inception的构造方法，只控制卷积核的个数，卷积核大小已经写好
        conv2d3x3red，conv2d5x5red是避免与conv2d1x1重名
        :param conv2d1x1: 1x1的卷积
        :param conv2d3x3red: 1x1卷积 - 降维
        :param conv2d3x3: 3x3卷积（有Padding，不会减小特征层大小）
        :param conv2d5x5red: 1x1卷积 - 降维
        :param conv2d5x5: 5x5卷积（有Padding，不会减小特征层大小）
        :param pool_pro: 1x1的卷积 - 同样是降维
        :param kwargs: 可变长度字典、方便传入层名称
        """
        super(Inception, self).__init__(**kwargs)
        # 这里第一个1x1的卷积因为没有用SAME方式，所以尺寸会缩小
        self.branch1 = layers.Conv2D(conv2d1x1, kernel_size=1, activation='relu')

        # 第二个是先用1x1的卷积核降维，然后再用'SAME'方法可以使得和branch1一样，
        # 这样就能使得卷积后的图像和branch1保持同样的shape
        self.branch2 = models.Sequential([
            layers.Conv2D(conv2d3x3red, kernel_size=1, activation='relu'),
            layers.Conv2D(conv2d3x3, kernel_size=3, padding='SAME', activation='relu')
        ])

        # branch3同理
        self.branch3 = models.Sequential([
            layers.Conv2D(conv2d5x5red, kernel_size=1, activation='relu'),
            layers.Conv2D(conv2d5x5, kernel_size=5, padding='SAME', activation='relu')
        ])

        # 先池化、再卷积
        self.branch4 = models.Sequential([
            layers.MaxPool2D(pool_size=3, strides=1, padding='SAME'),
            layers.Conv2D(pool_pro, kernel_size=1, activation='relu')
        ])

    def call(self, inputs, **kwargs):
        """
        call方法通俗的说，应用于：类创建之后，（）用作为方法调用如
        a = A()
        b = a(c)
        :param inputs:
        :param kwargs:
        :return:
        """
        branch1 = self.branch1(inputs)
        branch2 = self.branch2(inputs)
        branch3 = self.branch3(inputs)
        branch4 = self.branch4(inputs)
        # 将四个输出在深度方向进行拼接，
        outputs = layers.concatenate([branch1, branch2, branch3, branch4])

        return outputs


class InceptionAux(layers.Layer):
    """
    Inception辅助分类器
    """
    def __init__(self, num_class, **kwargs):
        super(InceptionAux, self).__init__(**kwargs)
        self.averagePool = layers.AvgPool2D(pool_size=5, strides=3, padding='VALID')
        self.conv = layers.Conv2D(128, kernel_size=1, activation="relu")

        self.fc1 = layers.Dense(1024, activation='relu')
        self.fc2 = layers.Dense(num_class)
        self.softmax = layers.Softmax()

    def call(self, inputs, **kwargs):
        x = self.averagePool(inputs)
        x = self.conv(x)
        x = layers.Flatten()(x)
        x = layers.Dropout(rate=0.5)(x)       # 原论文是70%但50%会好点
        x = self.fc1(x)
        x = layers.Dropout(rate=0.5)(x)
        x = self.fc2(x)
        x = self.softmax(x)

        return x


def GoogLeNet(height, width, class_num, channel, aux_logits=False):
    input_image = layers.Input(shape=(height, width, channel), dtype="float32")
    # (None, 224, 224, 3)
    x = layers.Conv2D(64, kernel_size=7, strides=2, padding='SAME', activation='relu', name='conv2d_1')(input_image)
    # (None, 112, 112, 64)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_1')(x)

    # (None, 56, 56, 64)
    x = layers.Conv2D(64, kernel_size=1, strides=1, activation='relu', name='conv2d_2')(x)
    # (None, 56, 56, 64)
    x = layers.Conv2D(192, kernel_size=3, strides=1, padding='SAME', activation='relu', name='conv2d_3')(x)
    # (None, 56, 56, 192)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_2')(x)

    # (None, 28, 28, 192)
    x = Inception(64, 96, 128, 16, 32, 32, name='inception_3a')(x)
    # (None, 28, 28, 256=64+128+32+32)
    x = Inception(128, 128, 192, 32, 96, 64, name='inception_3b')(x)
    # (None, 28, 28, 480)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_3')(x)

    # (None, 14, 14, 480)
    x = Inception(192, 96, 208, 16, 48, 64, name='inception_4a')(x)
    # 辅助分类器1
    if aux_logits:
        aux1 = InceptionAux(class_num, name='aux_1')(x)

    # (None, 14, 14, 512)
    x = Inception(160, 112, 224, 24, 64, 64, name='inception_4b')(x)
    # (None, 14, 14, 512)
    x = Inception(128, 128, 256, 24, 64, 64, name='inception_4c')(x)
    # (None, 14, 14, 512)
    x = Inception(112, 144, 288, 32, 64, 64, name='inception_4d')(x)
    # 辅助分类器2
    if aux_logits:
        aux2 = InceptionAux(class_num, name='aux_2')(x)

    # (None, 14, 14, 528)
    x = Inception(256, 160, 320, 32, 128, 128, name='inception_4e')(x)
    # (None, 14, 14, 832)
    x = layers.MaxPool2D(pool_size=3, strides=2, padding='SAME', name='maxpool_4')(x)

    # (None, 7, 7, 832)
    x = Inception(256, 160, 320, 32, 128, 128, name='inception_5a')(x)
    # (None, 7, 7, 832)
    x = Inception(384, 192, 384, 48, 128, 128, name='inception_5b')(x)
    # (None, 7, 7, 1024)
    x = layers.AvgPool2D(pool_size=7, strides=1, name='avgpool')(x)
    # (None, 1, 1, 1024)
    x = layers.Flatten(name='output_flatten')(x)
    x = layers.Dropout(rate=0.4, name='output_dropout')(x)
    # (None * 1024)
    x = layers.Dense(class_num, activation='relu')(x)
    # (None, class_num)
    output = layers.Softmax()(x)
    if aux_logits:
        model = models.Model(inputs=input_image, outputs=[aux1, aux2, output])
    else:
        model = models.Model(inputs=input_image, outputs=output)

    model.summary()

    return model


def model_train(model, x_train, x_val, epochs, train_step, val_step, weights_path, optimizer):
    """
    模型训练
    :param model: 定义好的模型
    :param x_train: 训练集数据
    :param x_val: 验证集数据
    :param epochs: 迭代次数
    :param train_step: 一个epoch的训练次数
    :param val_step: 一个epoch的验证次数
    :param weights_path: 权值保存路径
    :param optimizer: 优化器
    :return: None
    """
    best_test_loss = float('inf')
    # 将数据集实例化成迭代器
    train_datasets = iter(x_train)
    val_datasets = iter(x_val)

    for epoch in range(1, epochs + 1):
        # 清理历史信息
        train_loss.reset_states()       
        train_accuracy.reset_states()   
        test_loss.reset_states()        
        test_accuracy.reset_states()

        train_losses = []
        train_acc = []
        val_losses = []
        val_acc = []

        # 计算训练集集
        process_bar = tqdm(range(train_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            images, labels = next(train_datasets)
            with tf.GradientTape() as tape:
                aux1, aux2, output = model(images, training=True)
                loss1 = loss_object(labels, aux1)
                loss2 = loss_object(labels, aux2)
                loss3 = loss_object(labels, output)
                # 这2个0.3是根据论文提出的
                loss = loss1 * 0.3 + loss2 * 0.3 + loss3

            # 反向传播梯度下降
            gradients = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))

            train_loss(loss)
            train_accuracy(labels, output)
            train_losses.append(train_loss.result())
            train_acc.append(train_accuracy.result())

            process_bar.set_postfix({'loss': '{:.5f}'.format(train_loss.result()),
                                     'acc': '{:.5f}'.format(train_accuracy.result() * 100)})

        # 输出平均后的训练结果
        print("\rtrain_loss:{:.4f} train_accuracy:{:.4f}".format(np.mean(train_losses), np.mean(train_acc)))

        # 计算验证集
        process_bar = tqdm(range(val_step), ncols=100, desc="Epoch {}".format(epoch), unit="step")
        for _ in process_bar:
            images, labels = next(val_datasets)
            # 验证集不需要参与到训练中，因此不需要计算梯度
            _, _, output = model(images, training=False)
            loss = loss_object(labels, output)

            test_loss(loss)
            test_accuracy(labels, output)
            val_losses.append(test_loss.result())
            val_acc.append(test_accuracy.result())

            process_bar.set_postfix({'loss': '{:.5f}'.format(test_loss.result()),
                                     'acc': '{:.5f}'.format(test_accuracy.result() * 100)})

        # 输出平均后的验证结果
        print("\rvalidation_loss:{:.4f} validation_accuracy:{:.4f}\n".format(np.mean(val_losses), np.mean(val_acc)))

        if test_loss.result() < best_test_loss:
            best_test_loss = test_loss.result()
            model.save_weights(weights_path)


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
    img_path = './dataset/daisy.jpg'

    # 值得一提的是，这里开启图片如果用其他方式，需要考虑读入图片的通道数，在制作训练集时采用的是RGB，而opencv采用的则是BGR
    image = tf.io.read_file(img_path)
    image = tf.image.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [height, width])

    # 输入到网络必须是一个batch(batch_size, height, weight, channels)
    # 用这个方法去扩充一个维度
    image = (np.expand_dims(image, 0))

    model.load_weights(weights_path)

    # 这里不需要辅助分类器的值
    _, _, output = model.predict(image)
    # 预测的结果是包含batch这个维度，所以要把这个batch这维度给压缩掉
    result = np.squeeze(output)
    predict_class = int(np.argmax(result))
    print("预测类别：{}, 预测可能性{:.03f}".format(class_indict[predict_class], result[predict_class]*100))


def main():
    dataset_path = './dataset/'
    train_dir = os.path.join(dataset_path, 'train')
    val_dir = os.path.join(dataset_path, 'validation')
    weights_path = "./logs/weights/GooLeNet.h5"

    width = height = 224
    channel = 3

    batch_size = 32
    num_classes = 5
    epochs = 30
    lr = 0.0003
    is_train = False

    # 选择编号为0的GPU
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

    model = GoogLeNet(height, width, num_classes, channel, aux_logits=True)
    # 可以选择载入上次的权重，继续训练
    # model.load_weights(weights_path)

    optimizer = optimizers.Adam(learning_rate=lr)

    if is_train:
        # 模型训练
        model_train(model, train_dataset, val_dataset, epochs, train_step, val_step, weights_path, optimizer)
    else:
        # 模型预测
        model_predict(model, weights_path, height, width)


if __name__ == "__main__":
    # 自定义损失、优化器、准确率
    loss_object = losses.CategoricalCrossentropy(from_logits=False)

    train_loss = metrics.Mean(name='train_loss')
    train_accuracy = metrics.CategoricalAccuracy(name='train_accuracy')

    # 自定义损失和准确率方法
    test_loss = metrics.Mean(name='test_loss')
    test_accuracy = metrics.CategoricalAccuracy(name='test_accuracy')

    main()
