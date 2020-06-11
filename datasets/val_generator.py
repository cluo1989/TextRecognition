# coding: utf-8

import os
import cv2
import numpy as np
from tensorflow import keras
from charset import alphabet

# validate dataset 32*320, 10 chars
time_step = 320/4
word_length = 10


def char2index(word):
    indexs = []
    for c in word:
        try:
            index = alphabet.index(c)
        except:
            print('Index failed:', word)
        indexs.append(index)
                
    return indexs

def normalization(img):
    img = np.float32(img)
    img = img / 127.5
    img -= 1
    return img
    
def read_prepro_image(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # prepro
    img = normalization(img)

    return img

def get_batch(imglist, labellist):
    '''
    parse image dataset.
    label_file, content format: 'image_name.jpg label'
    '''
    assert len(imglist) == len(labellist), print("len of imglist and labellist not equal")
    imageData = []
    labelData = []

    for i in range(len(imglist)):
        # char2index
        indexes = char2index(labellist[i])
        labelData.append(indexes)

        # read and prepro image
        image = read_prepro_image(imglist[i])
        imageData.append(image)

    imageData = np.array(imageData)
    imageData = np.expand_dims(imageData, -1)
    labelData = np.array(labelData)

    n = imageData.shape[0]
    input_length = np.ones([n, 1]) * time_step
    label_length = np.ones([n, 1]) * word_length

    y_output = np.zeros([n, 1])
    x_input = [imageData, labelData, input_length, label_length]

    return x_input, y_output


class ValDataGenerator(keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, imglist, labellist, batch_size):
        'Initialization'
        self.imglist = imglist
        self.labellist = labellist
        self.batch_size = batch_size
        
    def __len__(self):
        'The number of batches per epoch'
        return int(np.ceil(len(self.imglist)/float(self.batch_size)))

    def __getitem__(self, index):
        'Generate one batch of data'
        imglist = self.imglist[index*self.batch_size:(index+1)*self.batch_size]
        labellist = self.labellist[index*self.batch_size:(index+1)*self.batch_size]
        x_input, y_output = get_batch(imglist, labellist)

        return x_input, y_output

    def on_epoch_end(self):
        'Modify your dataset between epochs,'
        'Do something like "shuffle indexes"'
        # generator_data.save_char_statistic()
        pass
