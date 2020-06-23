# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D, Activation, AveragePooling2D
from tensorflow.keras.layers import Bidirectional, Input, Permute, TimeDistributed, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, LSTM
from tensorflow.keras.regularizers import l2
import tensorflow as tf

# import config
import sys
sys.path.append("..")
from nets.resnet import resnet_50
from nets.residual_block import BottleNeck
from charset import alphabet

K.clear_session()
n_classes = len(alphabet) + 1  # 1000

hidden_dim = 256


def ctc_decode_func(y_pred):
    '''
    y_pred: shape (samples, timesteps, num_categories)
    return decoded results, List or Tuple.
    '''
    seq_len = K.shape(y_pred)[1]               # Time_steps
    batch_size = K.shape(y_pred)[0]            # Num_samples

    seq_len = tf.cast(seq_len, 'float32')
    seq_len = tf.ones((batch_size,)) * seq_len # 每个样本的序列长度

    result = K.ctc_decode(y_pred, seq_len)[0]

    return result

def ctc_loss_func(args):
    '''
    compute CTC Loss on each batch element.
    '''
    y_pred, labels, input_length, label_length = args

    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)

def cnn_baseline(input):
    '''
    original (paper setting) backbone CNN of crnn.
    原文 BatchNormalization 只在 layer4 出现
    MaxPooling 设置是否正确? x 默认为NHWC, 所以应该正确
    '''
    # layer1
    x = Conv2D(64, (3, 3), strides=(1, 1), padding="same", name="conv1")(input)
    #x = BatchNormalization(name="conv1_bn")(x)
    x = Activation("relu")(x)                 # 可以在 Conv2D 中指定 (如果Conv2D直接Activate)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)  # 可简写为 MaxPool2D(2)(x), default padding="valid"

    # layer2
    x = Conv2D(128, (3, 3), strides=(1, 1), padding="same", name="conv2")(x)
    #x = BatchNormalization(name="conv2_bn")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((2, 2), strides=(2, 2))(x)

    # layer3
    x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3_1")(x)
    #x = BatchNormalization(name="conv3_1_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(256, (3, 3), strides=(1, 1), padding="same", name="conv3_2")(x)
    #x = BatchNormalization(name="conv3_2_bn")(x)
    x = Activation("relu")(x)
    
    x = MaxPool2D((2, 2), strides=(2, 1))(x)

    # layer4
    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv4_1")(x)
    x = BatchNormalization(name="conv4_1_bn")(x)
    x = Activation("relu")(x)

    x = Conv2D(512, (3, 3), strides=(1, 1), padding="same", name="conv4_2")(x)
    x = BatchNormalization(name="conv4_2_bn")(x)
    x = Activation("relu")(x)

    x = MaxPool2D((2, 2), strides=(2, 1))(x)

    # layer5
    x = Conv2D(512, (2, 2), strides=(1, 1), name="conv5")(x)  # default padding="valid"
    x = Activation("relu")(x)

    return x

def cnn_resnet(input):
    '''
    backbone for crnn, based on resnet architecture.
    h stride=2*5, w stride=2*2
    '''
    x = Conv2D(32, (7, 7), strides=(1, 1), padding="same", name="conv1")(input)  # original: filters=64, strides=(2, 2)
    x = BatchNormalization(name="conv1_bn")(x)
    x = Activation("relu")(x)
    x = MaxPool2D((3, 3), strides=(2, 2))(x)

    inplane = 32  # 64 
    layer_nums = [3, 4, 6, 3]  # [2, 3, 5, 2]
    filters = [32, 64, 128, 256]  # [64, 128, 256, 512]
    stride_lists = [(2, 1), (2, 2), (2, 1), (2, 1)]  # First Layer don't Downsample, 

    for i in range(4):
        x = BottleNeck(inplane, filters[i], stride=stride_lists[i])(x)  # First Block of Each Layer do Downsample (except 1st Layer here)
        inplane = filters[i] * BottleNeck.expansion
        for _ in range(1, layer_nums[i]):
            x = BottleNeck(inplane, filters[i], stride=1)(x)

    return x

def cnn_resnet_stn(input):
    '''
    backbone for crnn, based on resnet and stn.
    '''
    return

def crnn(phase='train'):
    input_shape = (32, None, 1)  # HWC, H fixed
    img_input = Input(shape=input_shape, name="inputs")

    # Convolutional Layers
    # x = cnn_baseline(img_input)
    x = cnn_resnet(img_input)

    # Map to Sequence
    x = Permute((2, 1, 3))(x)  # 转为 WHC
    x = TimeDistributed(Flatten())(x)  # 将 Op(这里是Flatten) 应用于输入的每个时间片, 输入的第一维是时间维度

    # Recurrent Layers
    x = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.25, name="lstm1"))(x)
    x = Bidirectional(LSTM(hidden_dim, return_sequences=True, dropout=0.25, name="lstm2"))(x)
    y_pred = Dense(n_classes, activation="softmax")(x)

    # Transcription Layer
    # train & inference
    if phase is 'train':
        labels = Input(shape=[None, ], dtype="float32", name="labels")
        input_length = Input(shape=[1], dtype="int32", name="input_length")
        label_length = Input(shape=[1], dtype="int32", name="label_length")
        
        ctc_loss = Lambda(ctc_loss_func, output_shape=(1,), name="ctc")([y_pred, labels, input_length, label_length])
        model = Model(inputs=[img_input, labels, input_length, label_length], outputs=[ctc_loss])#y_pred, 
        
    else:
        decode_pred = Lambda(ctc_decode_func, name="ctc_decode")(y_pred)
        model = Model(inputs=img_input, outputs=decode_pred)
    
    return model

if __name__ == "__main__":
    model = crnn()
    model.summary()
