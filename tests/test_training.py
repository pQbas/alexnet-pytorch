import pytest
import alexnet

def test_training_train():
    CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
    alexnet.train(paramsPath = CONFIG_FILE_PATH)
