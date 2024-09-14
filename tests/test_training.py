import torch
import pytest
from alexnet import train

input = torch.tensor([[0.0993, 0.1001, 0.1005, 0.0999, 0.1008, 0.0998, 0.0992, 0.0994, 0.1007, 0.1004],
                      [0.1000, 0.0999, 0.1003, 0.1000, 0.1004, 0.0998, 0.0996, 0.0992, 0.1001, 0.1006],
                      [0.0996, 0.1003, 0.1008, 0.1000, 0.1005, 0.0996, 0.0995, 0.0991, 0.1001, 0.1005],
                      [0.0991, 0.0995, 0.1008, 0.1002, 0.1010, 0.0994, 0.1001, 0.0990, 0.0998, 0.1012]])


def test_training_testEpoch():
    CONFIG_FILE_PATH = '/test/alexnet/config.ini'

    params = train.getConfig(CONFIG_FILE_PATH) 

    device = train.getDevice()

    _, testset = train.getDataset(name = 'cifar')

    model = train.buildModel(numCategories = int(params['categories']),
                             device        = device)

    testloader = train.buildDataloader(testset,
                                       batchsize = int(params['batch_size']))
     
    acc = train.testEpoch(model, testloader, device)
        
    assert isinstance(acc, float), f"Expected float, got {type(acc)}"
    assert 0.0 <= acc <= 1.0, f"Accuracy out of bounds: {acc}"


