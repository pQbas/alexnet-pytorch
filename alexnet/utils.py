import torch
import torchvision
import torchvision.transforms as transforms
from alexnet.model import AlexNet
import configparser
import numpy as np
from pathlib import Path
import warnings


def getConfig(
    filePath : str
    ):
    print('file path:', filePath)
    config = configparser.ConfigParser()
    config.read(filePath)

    settings = {}

    for key, value in config['params'].items():
        if value.isdigit():
            settings[key] = int(value)

    for key, value in config['params'].items():
        try:
            settings[key] = float(value)
        except:
            pass

    true_values = ['true', 'yes']
    false_values = ['false', 'no']

    for key, value in config['params'].items():
        if value.lower() in true_values:
            settings[key] = True
        elif value.lower() in false_values:
            settings[key] = False

    for key, value in config['params'].items():
        if key not in settings:
            settings[key] = value

    return settings


def getDataset(
    name : str,
    path : str = './data'
):
    '''
    Returns a dataset
    '''
    trainset, testset = None, None
    
    if name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=path, 
                                                train=True,
                                                download=True, 
                                                transform=TRAIN_TRANSFORM)
        
        testset = torchvision.datasets.CIFAR10(root=path, 
                                               train=False,
                                               download=True, 
                                               transform=TRAIN_TRANSFORM)
    else:
        raise ValueError("At getDataset 'name' param must be one of valid datasets") 
    return (trainset, testset)


def getDevice() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def buildDataloader(
    dataset,
    batchsize  : int,
):
    '''
    Build a dataloader to the dataset with a batch
    -size and a typeLoader (train, test, val), for
    different use
    '''

    loader = torch.utils.data.DataLoader(dataset,
                                         batch_size=batchsize,
                                         shuffle=False,
                                         num_workers=2,
                                         sampler=np.random.permutation(10))
    return loader

def loadModel(
    weightsPath : str, 
    paramsPath  : str, 
    device      : str
    ):
    '''
    Loads the weights path to the model, and
    return the model
    '''
    params = getConfig(paramsPath)
    model = AlexNet(categories = int(params['categories']))
    model.load_state_dict(torch.load(weightsPath, map_location=device, weights_only=True))
    model.to(device)
    return model


TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

INFERENCE_TRANSFORM = transforms.Compose([ 
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def ensureDirectoryExists(dirPath):
    dirPath = Path(dirPath)
    
    # Check if the directory already exists
    if dirPath.is_dir():
        print(f"Directory '{dirPath}' already exists.")
        return

    # Check if the parent directory exists
    parentDir = dirPath.parent
    if not parentDir.is_dir():
        warnings.warn(f"Parent directory '{parentDir}' does not exist. Cannot create '{dirPath}'.")
        return

    # If the parent directory exists, create the new directory
    dirPath.mkdir(exist_ok=True)
    print(f"Directory '{dirPath}' has been created successfully.")

 

