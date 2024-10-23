import torch
import torchvision
import torchvision.transforms as transforms
from alexnet.model import AlexNet
import configparser
import numpy as np
from pathlib import Path
import warnings
import torch.nn as nn

import logging
logger = logging.getLogger(__name__)  # Get logger for this module


def getConfig(
    filePath : str
    ):
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

    elif name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=path, 
                                              train=True,
                                              download=True, 
                                              transform=TRAIN_TRANSFORM)
        
        testset = torchvision.datasets.MNIST(root=path, 
                                             train=False,
                                             download=True, 
                                             transform=TRAIN_TRANSFORM)

    elif name == 'fashion_mnist':

        trainset = torchvision.datasets.FashionMNIST(root=path, 
                                                     train=True,
                                                     download=True, 
                                                     transform=TRAIN_TRANSFORM)
        
        testset = torchvision.datasets.FashionMNIST(root=path, 
                                                    train=False,
                                                    download=True, 
                                                    transform=TRAIN_TRANSFORM)

    else:
        raise ValueError("At getDataset 'name' param must be one of valid datasets") 

    logger.info(f'Dataset loaded: {name},' 
                f'Training samples: {len(trainset)},' 
                f'Test samples: {len(testset)}')

    return (trainset, testset)


def getDevice() -> str: 
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
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
                                         num_workers=2)
                                        # sampler=np.random.permutation(200)

    logger.info(f'dataloaders created with batch size: {batchsize}')

    return loader


def buildModel(
        numCategories: int, 
        device: str
    ):
    '''
    Builds the model, moves it to the specified device, and performs a dummy forward pass.
    '''
    logger.debug(f"Building model for {numCategories} categories on device {device}")
    
    model = AlexNet(categories=numCategories).to(device)  # Assuming AlexNet is defined elsewhere
    dummyInput = torch.rand([1, 3, 224, 224]).to(device)
    model(dummyInput)  # Forward pass to initialize the model

    logger.info(f'Model created and initialized on device {device}')
    logger.debug("Model initialized successfully")
    return model



from typing import Optional, Dict

def loadModel(
    params      : Optional[Dict] = None,
    weightsPath : Optional[str]  = None, 
    paramsPath  : Optional[str]  = None, 
    device      : Optional[str]  = None
    ):
    '''
    Loads the weights path to the model, and
    return the model
    '''
    # Check that only one of the parameters is provided
    if paramsPath is not None and params is not None:
        raise ValueError("You can only use one of 'paramsPath' or 'params', not both.")
    
    if paramsPath is None and params is None:
        raise ValueError("You must provide either 'paramsPath' or 'params'.")

    # Load parameters if params is not directly provided
    if params is None:
        params = getConfig(paramsPath)
        logger.info(f'Loaded training parameters from {paramsPath}')
    else:
        logger.info('Using provided parameters directly.')
 
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


def ensureDirectoryExists(
    dirPath
    ):
    dirPath = Path(dirPath)
    
    # Check if the directory already exists
    if dirPath.is_dir():
        logger.info(f"Directory '{dirPath}' already exists.")
        return

    # Check if the parent directory exists
    parentDir = dirPath.parent
    if not parentDir.is_dir():
        warnings.warn(f"Parent directory '{parentDir}' does not exist. Cannot create '{dirPath}'.")
        return

    # If the parent directory exists, create the new directory
    dirPath.mkdir(exist_ok=True)
    logger.info(f"Directory '{dirPath}' has been created successfully.")


def buildLoss(
    typeLoss: str
    ):
    '''
    Returns the loss function based on the typeLoss parameter.
    '''
    logger.debug(f"Initializing loss function: {typeLoss}")

    lossf = None

    if typeLoss == 'binary-cross-entropy':
        lossf = nn.BCELoss()
    elif typeLoss == 'cross-entropy':
        lossf = nn.CrossEntropyLoss()
    else:
        logger.error(f"Invalid loss function type: {typeLoss}")
        raise ValueError("Invalid loss function type")

    logger.info(f'Loss initialized: {typeLoss}')

    return lossf


def buildOptimizer(model: nn.Module, typeOptimizer: str, learningRate: float):
    '''
    Returns the optimizer based on the typeOptimizer parameter.
    '''
    logger.debug(f"Initializing optimizer: {typeOptimizer} with learning rate: {learningRate}")

    optimizer = None

    if typeOptimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learningRate)
    else:
        logger.error(f"Invalid optimizer type: {typeOptimizer}")
        raise ValueError("Invalid optimizer type")

    logger.info(f'Optimizer initialized: {typeOptimizer}')
    return optimizer

