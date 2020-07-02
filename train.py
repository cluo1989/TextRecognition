# coding: utf-8
import os
from datetime import datetime
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler, TensorBoard
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import get_file
import tensorflow.keras.backend as K

import sys
sys.path.append("..")

from architects.crnn import crnn
from datasets.train_generator import TrainDataGenerator
from datasets.val_generator import ValDataGenerator

import config as cfg


def train(start_epoch, stop_epoch):
    
    # build model
    model = crnn('train')
    model.summary()

    # 标准 loss 函数必须定义为 loss(y_true, y_pred) 的形式, y_pred 为模型输出, 由于在 crnn.py 中
    # 已经定义了 outputs=[ctc_loss], 所以这里 y_pred 即为 ctc_loss, 故 loss 无需再计算, 直接返回就好了.
    # https://github.com/qqwweee/keras-yolo3/issues/481
    # https://stackoverflow.com/questions/51156885/what-is-y-pred-in-keras
    # https://github.com/keras-team/keras/blob/master/examples/image_ocr.py

    #sgd = SGD(learning_rate=0.001, decay=1e-6, momentum=0.9, nesterov=True)
    #adm = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)
    #adm = Adam(lr=1e-4)
    model.compile(optimizer='adam', 
                  loss={'ctc': lambda y_true, y_pred: y_pred},
                  )#metrics=["acc"]
    
    # get last weight
    path = os.path.join(cfg.OUTPUT_DIR, os.path.join('checkpoint'))
    files = os.listdir(path)
    if len(files) > 0 and start_epoch != 0:
        start_weights = "{:04d}".format(start_epoch)
        for p in files:
            if start_weights in p:
                weight_file = p
                model.load_weights(os.path.join(path, weight_file))
                print(os.path.join(path, weight_file),'===============')

    # captures output of softmax so we can decode the output during visualization
    #test_func = K.function(model.inputs, model.outputs)
    #viz_cb = VizCallback(run_name, test_func, img_gen.next_val())

    # callbacks
    lr_base = 1e-4
    def learning_rate_schedule(epoch):
        lr = lr_base * 0.9 ** int(epoch/10)            
        return lr

    lr_schedule = LearningRateScheduler(schedule=learning_rate_schedule)
    early_stopping = EarlyStopping(monitor='loss', patience=10)
    tensorboard = TensorBoard(log_dir='outputs/summary/'+datetime.now().strftime("%Y:%m:%d-%H:%M:%S"))

    class ParallelModelCheckpoint(ModelCheckpoint):
        def __init__(self, model, filepath, 
                    monitor='val_loss', 
                    verbose=0,
                    save_best_only=False, 
                    save_weights_only=False,
                    mode='auto', 
                    period=1):
            self.single_model = model
            super(ParallelModelCheckpoint, self).__init__(filepath, 
                                                          monitor, 
                                                          verbose, 
                                                          save_best_only, 
                                                          save_weights_only, 
                                                          mode, 
                                                          period)

        def set_model(self, model):
            super(ParallelModelCheckpoint, self).set_model(self.single_model)

    filepath = 'outputs/checkpoint/weights.{epoch:04d}-{loss:.4f}-{val_loss:.4f}.h5'
    model_checkpoint = ModelCheckpoint(filepath=filepath,
                                       monitor='val_loss',  # loss
                                       save_best_only=False,  # True,
                                       mode='min', 
                                       save_weights_only=True)
    # model_checkpoint = ParallelModelCheckpoint(model, 
    #                                            filepath, 
    #                                            monitor='val_loss',
    #                                            save_best_only=False,
    #                                            save_weights_only=True)

    callbacks = [lr_schedule, early_stopping, model_checkpoint, tensorboard]

    # get dataset generators
    batch_size = 32
    steps_per_epoch = 1000
    train_gen = TrainDataGenerator(batch_size, steps_per_epoch)

    valdir = "./datasets/validate_dataset/"
    label_file = valdir + "validate_dataset_labels.txt"
    f = open(label_file, "r")
    lines = f.readlines()
    
    imglist = []
    labellist = []
    for line in lines:
        imgname, label = line.strip().split('jpg ')
        imgname += "jpg"
        imgpath = valdir + imgname
        imglist.append(imgpath)
        labellist.append(label)

    val_gen = ValDataGenerator(imglist, labellist, batch_size)

    # start training
    model.fit_generator(generator=train_gen,
                        steps_per_epoch=steps_per_epoch,  # default: iterate over the training dataset
                        epochs=stop_epoch,
                        validation_data=val_gen,
                        validation_steps=5,  # default: iterate over the validation dataset
                        callbacks=callbacks,
                        initial_epoch=start_epoch,
                        max_queue_size=12,
                        use_multiprocessing=True,
                        workers=4
                        )


if __name__ == "__main__":
    train(1, 1000)
    # increase to wider images and start at epoch 20.
    # The learned weights are reloaded
    # train(20, 25)