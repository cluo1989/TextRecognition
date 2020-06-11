# coding: utf-8
import numpy as np
from tensorflow import keras
from datasets.text_renderer import generator_data


class TrainDataGenerator(keras.utils.Sequence):
    'Generates data for keras'
    def __init__(self, batch_size, batch_num, shuffle=True):
        'Initialization'
        self.batch_size = batch_size
        self.batch_num = batch_num
        self.shuffle = shuffle
        self.count_batch = 0

    def __len__(self):
        'The number of batches per epoch'
        return self.batch_num

    def __getitem__(self, index):
        'Generate one batch of data'
        x_input, y_output = generator_data.generator_batch(self.batch_size)
        self.count_batch += 1
        #print("current batch id:", self.count_batch)
        return x_input, y_output

    def on_epoch_end(self):
        'Do something like "shuffle indexes"'
        # generator_data.save_char_statistic()
        pass
