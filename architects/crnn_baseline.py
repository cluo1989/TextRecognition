# coding: utf-8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation, AveragePooling2D
from tensorflow.keras.layers import Bidirectional, Input, Permute, TimeDistributed, Lambda
from tensorflow.keras import backend as K
from tensorflow.keras.layers import BatchNormalization, LSTM, 
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers.merge import add
import tensorflow as tf

import config
from nets.resnet import resnet_50

K.clear_session()
n_classes = config.n_classes

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

def ctc_lambda_func(args):
    '''
    compute CTC Loss on each batch element.
    '''
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_loss(labels, y_pred, input_length, label_length)

def cnn_baseline(input):
    '''
    original (paper setting) backbone CNN of crnn.
    '''
    return

def cnn_resnet(input):
    '''
    backbone for crnn, based on resnet.
    '''
    return

def cnn_resnet_stn(input):
    '''
    backbone for crnn, based on resnet and stn.
    '''
    return

def crnn(phase='train'):
    input_shape = (32, None, 1)  # HWC, H fixed
    img_input = Input(shape=input_shape, name='input')

    # get cnn features
    x = cnn_baseline(img_input)

    # bilstm encode

    # ctc decode

    
