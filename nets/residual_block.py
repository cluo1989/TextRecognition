# coding: utf-8
from tensorflow import keras
from tensorflow.keras.layers import Layer, Conv2D, BatchNormalization, Activation


class BasicBlock(Layer):
    '''
    实现 BasicBlock 本不必设置 expansion, 初始化也不必 inplanes
    为了和 BottleNeck 保持调用方式一致, 才加入的
    '''
    expansion = 1

    def __init__(self, inplanes, filter_num, stride=1):
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
    expansion = 4

    def __init__(self, inplanes, filter_num, stride=1):
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
        self.conv3 = Conv2D(filters=filter_num * self.expansion,  # Block 输出做 channel expansion
                            kernel_size=(1, 1),
                            strides=1,
                            padding="same")
        self.bn3 = BatchNormalization()

        # downsample 用于 shortcut 中, 使得 shortcut 和 output 可相加 (HW,C 保持一致)
        # 对于第一个 Layer, 第一个 Block 不做空间 downsample (stride=1), 但 Block 输出和出入 channels 不一致, 仍需Conv1x1
        # 对于其余的 Layer, 第一个 Block 需要 downsample (stride!=1), 同时 Block 输出和出入 channels 不一致, 也需Conv1x1
        if stride != 1 or inplanes != filter_num * self.expansion:
            self.downsample = keras.Sequential()
            self.downsample.add(Conv2D(filters=filter_num * self.expansion,  # 使 channel 一致
                                    kernel_size=(1, 1),                      # 使用 Conv1x1
                                    strides=stride))                         # 使 spatial 一致
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
