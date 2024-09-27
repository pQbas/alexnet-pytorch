import os

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from datetime import datetime

from alexnet.model import AlexNet
# import alexnet.utils as utils
from alexnet.utils import getConfig, getDataset, getDevice, buildDataloader, loadModel, ensureDirectoryExists 
# getConfig, getDataset, getDevice, buildDataloader

import logging
logger = logging.getLogger(__name__)  # Get logger for this module

def saveModel(model: nn.Module, name: str, path: str):
    '''
    Saves model at the specified directory.
    Returns True if the model was saved successfully, False otherwise.
    '''
    ensureDirectoryExists(path)  # Assuming this function checks if the path exists
    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    modelName = f'{name}-{current_time}.pt'

    try:
        torch.save(model.state_dict(), os.path.join(path, modelName))
        logger.debug(f"Model '{modelName}' saved successfully at '{path}'")
        return True
    except Exception as e:
        logger.error(f"Error saving model '{modelName}' at '{path}': {e}")
        return False


def buildLoss(typeLoss: str):
    '''
    Returns the loss function based on the typeLoss parameter.
    '''
    logger.debug(f"Initializing loss function: {typeLoss}")
    
    if typeLoss == 'binary-cross-entropy':
        return nn.BCELoss()
    elif typeLoss == 'cross-entropy':
        return nn.CrossEntropyLoss()
    else:
        logger.error(f"Invalid loss function type: {typeLoss}")
        raise ValueError("Invalid loss function type")


def buildOptimizer(model: nn.Module, typeOptimizer: str, learningRate: float):
    '''
    Returns the optimizer based on the typeOptimizer parameter.
    '''
    logger.debug(f"Initializing optimizer: {typeOptimizer} with learning rate: {learningRate}")

    if typeOptimizer == 'sgd':
        return optim.SGD(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adam':
        return optim.Adam(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adadelta':
        return optim.Adadelta(model.parameters(), lr=learningRate)
    else:
        logger.error(f"Invalid optimizer type: {typeOptimizer}")
        raise ValueError("Invalid optimizer type")


def buildModel(numCategories: int, device: str):
    '''
    Builds the model, moves it to the specified device, and performs a dummy forward pass.
    '''
    logger.debug(f"Building model for {numCategories} categories on device {device}")
    
    model = AlexNet(categories=numCategories).to(device)  # Assuming AlexNet is defined elsewhere
    dummyInput = torch.rand([1, 3, 224, 224]).to(device)
    model(dummyInput)  # Forward pass to initialize the model
    
    logger.debug("Model initialized successfully")
    return model


def testEpoch(model, testLoader, device):
    '''
    Computes accuracy over the test set.
    '''
    total_correct, total_samples = 0, 0
    threshold = 0.5
    model.eval()
    logger.debug("Evaluating model on the test set")

    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        values, indices = torch.max(probs, dim=1)

        total_correct += torch.sum((indices == labels) * (values >= threshold)).item()
        total_samples += labels.size(0)

    accuracy = total_correct / total_samples
    logger.debug(f"Test accuracy: {accuracy:.4f}")
    return accuracy


def trainEpoch(model, trainLoader, optimizer, lossf, device):
    '''
    Trains the model for one epoch and returns the average loss.
    '''
    model.train()
    runningLoss = []
    logger.debug("Training model for one epoch")

    for data in trainLoader:
        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)
        
        loss = lossf(probs, labels)
        loss.backward()
        optimizer.step()

        runningLoss.append(loss.item())

    avg_loss = np.mean(runningLoss)
    logger.debug(f"Training epoch complete. Average loss: {avg_loss:.4f}")
    return avg_loss

from typing import Optional, Dict

def train(
    paramsPath : Optional[str]  = None,
    params     : Optional[Dict] = None
    ):
    logger.info('\n========== TRAINING ==========\n')

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
   
    device = getDevice()  # Assuming getDevice is defined elsewhere
    logger.info(f'Using device: {device}')

    # Log training configuration details
    logger.info(f'Training configuration | Model: {params["model_name"]} | Dataset: {params["dataset_name"]} | '
                f'Epochs: {params["epochs"]} | Batch size: {params["batch_size"]} | '
                f'Optimizer: {params["optimizer"]} | Learning rate: {params["learning_rate"]} | Loss function: {params["loss"]}')

    # Load datasets
    trainset, testset = getDataset(name=params['dataset_name'])  # Assuming getDataset is defined elsewhere
    logger.info(f'Dataset loaded: {params["dataset_name"]}, Training samples: {len(trainset)}, Test samples: {len(testset)}')

    # Build dataloaders
    trainloader = buildDataloader(trainset, batchsize=int(params['batch_size']))  # Assuming buildDataloader is defined elsewhere
    testloader = buildDataloader(testset, batchsize=int(params['batch_size']))
    logger.info(f'Dataloaders created with batch size: {params["batch_size"]}')

    # Build model
    model = buildModel(numCategories=int(params['categories']), device=device)
    logger.info(f'Model {params["model_name"]} created and initialized on device {device}')

    # Build optimizer and loss function
    optimizer = buildOptimizer(model, typeOptimizer=params['optimizer'], learningRate=params['learning_rate'])
    lossf = buildLoss(typeLoss=params['loss'])
    logger.info(f'Optimizer and loss function initialized: Optimizer: {params["optimizer"]}, '
                f'Learning rate: {params["learning_rate"]}, Loss function: {params["loss"]}')

    # Training loop
    for epoch in range(int(params['epochs'])):
        # logger.info(f'Epoch {epoch + 1}/{int(params["epochs"])} started')

        # Train the model for one epoch
        loss = trainEpoch(model, trainloader, optimizer, lossf, device)
        
        # Evaluate the model on the test set
        acc = testEpoch(model, testloader, device)

        logger.info(f'Epoch {epoch + 1}/{int(params["epochs"])} completed | Training Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')

    # Save the model after training
    if saveModel(model, name=params['model_name'], path=params['path']):
        logger.info(f'Model saved successfully as {params["model_name"]} at {params["path"]}')
    else:
        logger.error(f'Failed to save model {params["model_name"]} at {params["path"]}')

    # Clean up GPU memory if necessary
    torch.cuda.empty_cache()
    logger.info('Training finished, GPU memory cache cleared')

    return
