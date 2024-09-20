import pytest
import alexnet

def test_testing_main():
    CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
    WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/runs/alexnet.pt'
   
    accuracy = alexnet.test(paramsPath = CONFIG_FILE_PATH, weightsPath = WEIGHTS_PATH)

    assert isinstance(accuracy, float), f"Expected float, got {type(accuracy)}"
    assert 0.0 <= accuracy <= 1.0, f"Accuracy out of bounds: {accuracy}"


