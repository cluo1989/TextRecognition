# coding: utf-8
import os
import datetime
# from tensorflow.keras.callbacks import EarlyStopping, ModelCheckPoint, LearningRateScheduler
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K

import sys
sys.path.append("..")
from architects.crnn import crnn
from datasets import train_generator, val_generator
from datasets.generator import TextImageGenerator
from utils.visualize import VizCallback

import config as cfg


def train(run_name, start_epoch, stop_epoch, img_w):
    # Input Parameters
    img_h = 32
    words_per_epoch = 16000
    val_split = 0.2
    val_words = int(words_per_epoch * (val_split))
    minibatch_size = 32
    pool_size = 2

    # create generator
    fdir = os.path.dirname(
        get_file('wordlists.tgz',
                 origin='http://www.mythic-ai.com/datasets/wordlists.tgz',
                 untar=True))

    img_gen = TextImageGenerator(
        monogram_file=os.path.join(fdir, 'wordlist_mono_clean.txt'),
        bigram_file=os.path.join(fdir, 'wordlist_bi_clean.txt'),
        minibatch_size=minibatch_size,
        img_w=img_w,
        img_h=img_h,
        downsample_factor=(pool_size ** 2),
        val_split=words_per_epoch - val_words)
    
    # build model
    model = crnn('train')
    model.summary()

    sgd = SGD(learning_rate=0.02, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=sgd)
    # https://stackoverflow.com/questions/51156885/what-is-y-pred-in-keras
    # https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

    if start_epoch > 0:
        weight_file = os.path.join(
            cfg.OUTPUT_DIR,
            os.path.join(run_name, 'weights%02d.h5' % (start_epoch - 1)))
        model.load_weights(weight_file)
        
    # captures output of softmax so we can decode the output during visualization
    test_func = K.function(model.inputs, model.outputs)
    viz_cb = VizCallback(run_name, test_func, img_gen.next_val())
    
    # start training
    model.fit_generator(generator=img_gen.next_train(),
                        steps_per_epoch=(words_per_epoch - val_words) // minibatch_size,
                        epochs=stop_epoch,
                        validation_data=img_gen.next_val(),
                        validation_steps=val_words // minibatch_size,
                        callbacks=[viz_cb, img_gen],
                        initial_epoch=start_epoch
                        )


if __name__ == "__main__":
    run_name = datetime.datetime.now().strftime('%Y:%m:%d:%H:%M:%S')
    train(run_name, 0, 20, 160)
    # increase to wider images and start at epoch 20.
    # The learned weights are reloaded
    train(run_name, 20, 25, 512)