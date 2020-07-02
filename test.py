from architects.crnn import crnn
from charset import alphabet
import numpy as np
import cv2
import os

# load model
model = crnn('test')
model.load_weights('outputs/checkpoint/weights.0012-0.0690-0.0025.h5')

# get validate list
valdir = 'datasets/validate_dataset/'
with open(valdir+'validate_dataset_labels.txt') as f:
    lines = f.readlines()

# test
cnt = 0
for idx, line in enumerate(lines):
    image, label = line.strip().split('jpg ')
    image += 'jpg'
    imgpath = valdir + image

    # read img
    img = cv2.imread(imgpath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.expand_dims(img, -1)

    # prepro
    # resize
    # normalization
    img = np.float32(img)
    img /= 127.5
    img -= 1

    # predict
    img = np.expand_dims(img, 0)
    indexs = model.predict(img)

    # index to char
    chars = [alphabet[i] for i in indexs[0]]
    result = ''.join(chars)
    if result == label:
        cnt += 1
    else:
        print(idx, result, label)

print('acc:{:.4f}'.format(cnt/len(lines)))