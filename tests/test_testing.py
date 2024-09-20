import pytest
from alexnet import test


def test_testing_main():
    CONFIG_FILE_PATH = '/test/alexnet/config.ini'
    WEIGHTS_PATH = '/test/runs/alexnet.pt'
    
    accuracy = test.test(paramsPath = CONFIG_FILE_PATH, weightsPath = WEIGHTS_PATH)

    assert isinstance(accuracy, float), f"Expected float, got {type(accuracy)}"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy out of bounds: {accuracy}"


