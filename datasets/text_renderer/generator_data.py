# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 13:27:55 2019

@author: hu
"""
import traceback
import numpy as np
import random
import multiprocessing as mp
from multiprocessing import Manager
from itertools import repeat
import os
import cv2
import concurrent.futures

import charset
from tenacity import retry
from datasets.text_renderer.para import GenPara
from datasets.text_renderer.image_transform import ImageDataGenerator_my

process_probility = 0.5
stretch_ratio = 1.0
batch_idx = 0
charset = charset.alphabet  # tongyong_charset
lock = mp.Lock()
counter = mp.Value('i', 0)
STOP_TOKEN = 'kill'

gen_para = GenPara()
datagen = ImageDataGenerator_my(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=True,  # set each sample mean to 0
        samplewise_std_normalization=False,  # divide each input by its std
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=1,  # randomly rotate images in the range (degrees, 0 to 180)
        width_shift_range=0.01,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        zoom_range=0.2)

statistic = {c:0 for c in list(charset)}

def stretch_compress_img(img):
    if stretch_ratio == 1.0:
        return img
    h, w = img.shape[0:2]
    w1 = int(w * stretch_ratio)
    resize_img = cv2.resize(img, (w1, h), interpolation=cv2.INTER_LINEAR)
    return resize_img


@retry
def gen_img_retry(renderer, img_index):
    try:
        im, word = renderer.gen_img(img_index)

    except Exception as e:
        print("Retry gen_img: %s" % str(e))
        traceback.print_exc()
        raise Exception

    if (word == '' or im is None or im.size == 0):
        print("Find gen_img return None or empty")
        raise Exception

    return im, word

def normalization_img(img):
    img = np.float32(img)
    # method1: 0~1
    img /= 127.5
    img -= 1

    # # method2: N(0, 1)
    # me = np.mean(img)
    # st = np.std(img) + 0.000001
    # img = (img - me) / st

    return img


def char_to_index(word):
    """ change str to index in charset
    """
    global statistic

    indexs = []
    for cc in word:
        try:
            index = charset.index(cc)
        except:
            print('-----', word)
        indexs.append(index)
        statistic[cc] += 1
        
    return indexs


def generator_data(img_index, q=None):
    im, word = gen_img_retry(gen_para.renderer, img_index)
    indexs = char_to_index(word)
    #print(word, indexs, charset[indexs[0]], charset[indexs[-1]])

    # add extra degradation
    # np.random.seed()
    # if np.random.random() > 0.5:
    #     im = datagen.flow(np.expand_dims(im, -1))

    # # save (debug)
    # name = '{0}_{1}_{2}'.format(batch_idx, img_index, word.replace('/', '_'))
    # cv2.imwrite('./datasets/text_renderer/generate_img/'+ name + '.jpg', im)

    # normalize
    im = normalization_img(im)

    return im, indexs
    # q.put((im, indexs))
    # print(q.qsize())


def multi_process(batch_img, batch_label, batch_size):
    global gen_para
    gen_para.initial_genpara()
    #print("+"*20, "into")
    # manager = mp.Manager()
    # q = manager.Queue()
    # with mp.Pool(processes=2) as pool:
    #     pool.starmap(generator_data, zip(range(batch_size), repeat(q)))
    #     q.put(STOP_TOKEN)
    
    # for i in range(batch_size):
    #     img, label = q.get()
    #     batch_img.append(img)
    #     batch_label.append(label)
    #print("--1--")
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:  # max_workers=4, default: os.cpu_count()
        #print("---")
        results = [executor.submit(generator_data, idx) for idx in range(batch_size)]
        #print("111", len(results), results[0].result())
        for res in concurrent.futures.as_completed(results):
            #print("222")
            img, label = res.result()
            #print("333")
            batch_img.append(img)
            batch_label.append(label)
        #print("+++")

    return batch_img, batch_label


def single_process(batch_img, batch_label, batch_size):
    global gen_para
    gen_para.initial_genpara()
    for i in range(batch_size):
        img, label = generator_data(i)
        #img = stretch_compress_img(img)

        batch_img.append(img)
        batch_label.append(label)

    return  batch_img, batch_label


def generator_batch(batch_size):
    global stretch_ratio
    global batch_idx

    while 1:
        # random stretch ratio
        if np.random.random() > 0.8:
            stretch_ratio = np.random.uniform(0.7, 1.2)
        else:
            stretch_ratio = 1.0

        batch_img = []
        batch_label = []
        #print("start get a batch ========")
        batch_img, batch_label = single_process(batch_img, batch_label, batch_size)
        #print("get a batch ====================================")

        batch_img = np.array(batch_img)
        batch_img = np.expand_dims(batch_img, -1)
        batch_label = np.array(batch_label)
        
        if batch_img.shape[0] != batch_size:
            print(batch_img.shape, 10*'+1+')
            continue

        if len(batch_img.shape) != 4:
            print(batch_img.shape, 10*'+2+')
            continue

        time_step = batch_img.shape[2] // (2**2)  # //4 - 3
        input_length = np.ones([batch_size, 1]) * time_step
        label_length = np.ones([batch_size, 1]) * batch_label.shape[1]
        x_input = [batch_img, batch_label, input_length, label_length]
        y_output = np.zeros([batch_size, 1])
        
        batch_idx += 1
        
        return x_input, y_output

def save_char_statistic():
    with open('statistics.txt', 'w') as f:
        print(statistic, file=f)

if __name__ == '__main__':
    #gen = generator_batch(20)
    labels = []
    i = 0
    # for i in range(10):
    while True:
        i += 1        
        res = generator_batch(20)#next(gen)
        #labels.append(res[0][1])
        print('----------------------', i, '   ', res[0][0].shape)
        #$print('----------------------', res[0][1][1:10])

    #print(statistic)
    with open('statistics.txt', 'w') as f:
        print(statistic, file=f)
