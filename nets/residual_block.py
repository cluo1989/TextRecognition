# coding: utf-8
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation


class BasicBlock(Layer):
    
    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = Conv2D(filters=filter_num, 
                            kernel_size=(3, 3),
                            strides=stride,  # 有可能 downsample
                            padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num, 
                            kernel_size=(3, 3),
                            strides=1,  # 不做 downsample
                            padding="same")
        self.bn2 = BatchNormalization()

        if stride != 1:
            self.downsample = keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num,
                                       kernel_size=(1, 1),
                                       strides=stride))
            self.downsample.add(BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        output = Activation('relu')(keras.layers.add([residual, x]))

        return output


class BottleNeck(Layer):

    def __init__(self, filter_num, stride=1):
        super(BottleNeck, self).__init__()
        self.conv1 = Conv2D(filters=filter_num, 
                            kernel_size=(1, 1),
                            strides=1,
                            padding="same")
        self.bn1 = BatchNormalization()
        self.conv2 = Conv2D(filters=filter_num, 
                            kernel_size=(3, 3),
                            strides=stride,  # 可能 downsample
                            padding='same')
        self.bn2 = BatchNormalization()
        self.conv3 = Conv2D(filters=filter_num*4,
                            kernel_size=(1, 1),
                            strides=1,
                            padding="same")
        self.bn3 = BatchNormalization()

        # 各个 Layer 的第一个 Block, stride != 1
        # shortcut 需要 downsample, 同时 Layer 之间 channels 需要一致
        if stride != 1:
            self.downsample = keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num*4,  # 和 conv3 一致都是 filter_num*4
                                    kernel_size=(1, 1),
                                    strides=stride))          # 和 conv2 一致都需 downsample
            self.downsample.add(BatchNormalization())
        # 若非第一个 Block (即 stride == 1 时, shortcut 不需要调整 identity 即可)
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)  # pytorch 官方实现中的 identity

        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = Activation('relu')(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
        output = Activation('relu')(keras.layers.add([residual, x]))

        return output


# def make_basicblock_layer(filter_num, blocks, stride=1):
#     res_block = Sequential()
#     res_block.add(BasicBlock(filter_num, stride=stride))

#     for _ in range(1, blocks):
#         res_block.add(BasicBlock(filter_num, stride=1))

#     return res_block

# def make_bottleneck_layer(filter_num, blocks, stride=1):
#     res_block = Sequential()
#     res_block.add(BottleNeck(filter_num, stride=stride))

#     for _ in range(1, blocks):
#         res_block.add(BottleNeck(filter_num, stride=1))

#     return res_block

# references: 
# https://github.com/calmisential/TensorFlow2.0_ResNet/blob/master/models/residual_block.py
# http://lanbing510.info/2017/08/21/ResNet-Keras.html
# https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
# https://github.com/raghakot/keras-resnet/blob/master/tests/test_resnet.py
# https://github.com/raghakot/keras-resnet/blob/master/cifar10.py
