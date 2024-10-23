import os

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from datetime import datetime

from alexnet.model import AlexNet
# import alexnet.utils as utils
from alexnet.utils import getConfig, getDataset, getDevice, buildDataloader, loadModel, ensureDirectoryExists, buildModel, buildLoss, buildOptimizer, track_metric 
# getConfig, getDataset, getDevice, buildDataloader
from typing import Optional, Dict
import mlflow
import mlflow.pytorch
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
        return os.path.join(path,modelName)
    except Exception as e:
        logger.error(f"Error saving model '{modelName}' at '{path}': {e}")
        return None


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


def train(
    paramsPath : Optional[str]  = None,
    params     : Optional[Dict] = None,
    tracking_train : bool = False
    ):
    logger.info('========== TRAINING ==========\n')

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
      
    # Log training configuration details
    logger.info(f'Training configuration | Model: {params["model_name"]} |' 
                f'Dataset: {params["dataset_name"]} | Epochs: {params["epochs"]} |'
                f'Batch size: {params["batch_size"]} | Optimizer: {params["optimizer"]} |' 
                f'Learning rate: {params["learning_rate"]} | Loss function: {params["loss"]}')
     
    # Load datasets
    trainset, testset = getDataset(name=params['dataset_name'])
     
    trainloader = buildDataloader(trainset, 
                                  batchsize=int(params['batch_size']))
      
    testloader = buildDataloader(testset, 
                                 batchsize=int(params['batch_size']))
    
    # Build model
    model = buildModel(numCategories=int(params['categories']), 
                       device=device)

    # Build optimizer and loss function
    optimizer = buildOptimizer(model, 
                               typeOptimizer=params['optimizer'], 
                               learningRate=params['learning_rate'])

    lossf = buildLoss(typeLoss=params['loss'])

    if tracking_train:
        mlflow.set_experiment("experimento_condicional")
        mlflow.start_run()
    
    # Registrar hiperparámetros solo si el tracking está activo
    track_metric(tracking_train, param_name="learning_rate", param_value = params['learning_rate'])
    track_metric(tracking_train, param_name="epochs", param_value = params['epochs'] )
    
    # Training loop
    for epoch in range(int(params['epochs'])):

        loss = trainEpoch(model, trainloader, optimizer, lossf, device)
        
        acc = testEpoch(model, testloader, device)

        track_metric(tracking_train, metric_name="loss", metric_value=loss, 
                     step=epoch)

        track_metric(tracking_train, metric_name="accuracy", metric_value=acc,
                     step=epoch)

        logger.info(f'Epoch {epoch + 1}/{int(params["epochs"])} completed |' 
                    f'Training Loss: {loss:.4f}, Test Accuracy: {acc:.4f}')


    modelPath = saveModel(model, name=params['model_name'], path=params['path'])  
    
    if modelPath:
        logger.info(f'Model saved successfully as {params["model_name"]} at {params["path"]}')
    else:
        logger.error(f'Failed to save model {params["model_name"]} at {params["path"]}')

    # Clean up GPU memory if necessary
    torch.cuda.empty_cache()
    logger.info('Training finished, GPU memory cache cleared')

    return modelPath
