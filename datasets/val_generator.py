# coding: utf-8

import os
import numpy as np
import cv2

time_step = 0# TODO??? 

def read_image(imgpath):
    img = cv2.imread()

    # prepro

def get_validate_dataset(imgdir, label_file):
    '''
    parse image dataset.
    label_file, content format: 'image_name.jpg label'
    '''
    imageData = []
    labelData = []

    with open(label_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            imgname, label = line.strip().split('jpg ')
            imgname += "jpg"

            # char2index
            indexes = char2index(label)
            labelData.append(indexes)

            # read and prepro image
            image = read_image(imgdir+"/" + imgname)
            imageData.append(image)

    imageData = np.array(imageData)
    labelData = np.array(labelData, dtype=np.float32)

    n = imageData.shape[0]
    input_length = np.ones([n, 1]) * time_step
    label_length = np.ones([n, 1]) * 10

    y_output = np.zeros([n, 1])
    x_input = [imageData, labelData, input_length, label_length]

    return x_input, y_output

