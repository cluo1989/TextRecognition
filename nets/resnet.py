# coding: utf-8
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, BatchNormalization, Activation, MaxPool2D, GlobalAveragePooling2D, Dense

import sys
sys.path.append("..")
from nets.residual_block import BasicBlock, BottleNeck


class ResNet(keras.Model):

    def __init__(self, block, layer_params, num_classes=1000):
        super(ResNet, self).__init__()
        self.inplanes = 64

        # Conv1
        self.conv1 = Conv2D(filters=self.inplanes,
                            kernel_size=(7, 7),
                            strides=2, 
                            padding="same")
        self.bn1 = BatchNormalization()
        self.pool1 = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")  # 这里完成第一次 downsample

        # 4 Layers, 每个由若干Block组成
        self.layer1 = self._make_layer(64, block, layer_params[0])
        self.layer2 = self._make_layer(128, block, layer_params[1], stride=2)
        self.layer3 = self._make_layer(256, block, layer_params[2], stride=2)
        self.layer4 = self._make_layer(512, block, layer_params[3], stride=2)

        # GAP & FC
        self.avgpool = GlobalAveragePooling2D()
        self.fc = Dense(units=num_classes, activation=keras.activations.softmax)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = Activation('relu')(x)
        x = self.pool1(x)

        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.avgpool(x)
        output = self.fc(x)

        return output

    def _make_layer(self, filter_num, block, block_num, stride=1):
        # 先来一个 block
        res_block = keras.Sequential()
        res_block.add(block(self.inplanes, filter_num, stride=stride))  # 每个 Layer 的第一个 Block 可能 downsample

        # 更新 Block 的输入 inplanes
        self.inplanes = filter_num * block.expansion  

        # 剩余 block
        for _ in range(1, block_num):                    # 剩余 Block 不做 downsample, stride=1
            res_block.add(block(self.inplanes, filter_num, stride=1))

        return res_block

def resnet_18(num_classes):
    return ResNet(BasicBlock, layer_params=[2, 2, 2, 2], num_classes=num_classes)

def resnet_34(num_classes):
    return ResNet(BasicBlock, layer_params=[3, 4, 6, 3], num_classes=num_classes)

def resnet_50(num_classes):
    return ResNet(BottleNeck, layer_params=[3, 4, 6, 3], num_classes=num_classes)

def resnet_101(num_classes):
    return ResNet(BottleNeck, layer_params=[3, 4, 23, 3], num_classes=num_classes)

def resnet_152(num_classes):
    return ResNet(BottleNeck, layer_params=[3, 8, 36, 3], num_classes=num_classes)

# TODO: 实现 ResNetV2 架构
