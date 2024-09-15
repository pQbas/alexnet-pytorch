import pytest
from alexnet import train

def test_training_train():
    CONFIG_FILE_PATH = '/test/alexnet/config.ini'
    train.train(paramsPath = CONFIG_FILE_PATH)
