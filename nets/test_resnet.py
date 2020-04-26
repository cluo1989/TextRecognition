# coding: utf-8
import pytest
from tensorflow.keras import backend as K
from resnet import resnet_34, resnet_50

DIM_ORDERING = {'channels_first', 'channels_last'}
IMG_HEIGHT = 224
IMG_WIDTH = 224
CHANNELS = 3

def _test_model_compile(model):
    for order in DIM_ORDERING:
        K.set_image_data_format(order)
        model.compile(loss='categorical_crossentropy', optimizer="sgd")
        assert True, "Failed to compile with {} dim ordering.".format(order)

def test_resnet34():
    model = resnet_34(10)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    model.summary()
    _test_model_compile(model)

def test_resnet50():
    model = resnet_50(10)
    model.build(input_shape=(None, IMG_HEIGHT, IMG_WIDTH, CHANNELS))
    model.summary()
    _test_model_compile(model)

if __name__ == "__main__":
    pytest.main([__file__])
