from alexnet.model import AlexNet
from alexnet.utils import getConfig, getDataset, getDevice, buildDataloader, loadModel

import torch
import torch.nn as nn
from tqdm import tqdm

CONFIG_PARAMS = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/runs/alexnet.pt'

import logging
logger = logging.getLogger(__name__)  # Get logger for this module

from typing import Optional, Dict

def test(
    params      : Optional[Dict] = None, 
    paramsPath  : Optional[str]  = None, 
    weightsPath : Optional[str]  = None
    ):
    logger.info('\n========== TESTING ==========\n')

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

    device = getDevice()
    logger.info(f'Using device: {device}')

    # Log test configuration details
    logger.info(f'Test configuration | Model Weights: {weightsPath} | Dataset: {params["dataset_name"]} | '
                f'Batch size: {params["batch_size"]}')

    # Load the test dataset
    _, testset = getDataset(name=params['dataset_name'])
    logger.info(f'Dataset {params["dataset_name"]} loaded with {len(testset)} test samples')

    # Create dataloader for the test set
    testLoader = buildDataloader(testset, batchsize=int(params['batch_size']))
    logger.info(f'Test dataloader created with batch size: {params["batch_size"]}')

    # Load the model with the provided weights
    print(params)
    if params is None:
        model = loadModel(weightsPath=weightsPath, paramsPath=paramsPath, device=device)
    else:
        model = loadModel(weightsPath=weightsPath, params=params, device=device)
    logger.info(f'Model loaded successfully from {weightsPath}')


    # Initialize metrics
    total_correct, total_samples = 0, 0

    # Set the model to evaluation mode
    threshold = 0.5
    model.eval()
    logger.info('Model set to evaluation mode')

    # Iterate through the test batches
    for data in testLoader:
        inputs, labels = data[0].to(device), data[1].to(device)

        # Perform forward pass and compute predictions
        outputs = model(inputs)
        probs = nn.Softmax(dim=1)(outputs)

        values, indices = torch.max(probs, dim=1)

        # Calculate the number of correct predictions
        total_correct += torch.sum((indices == labels) * (values >= threshold)).item()
        total_samples += labels.size(0)

        # Log progress for each batch (optional, depends on verbosity needed)
        logger.debug(f'Processed batch. Current total_correct: {total_correct}, total_samples: {total_samples}')

    # Compute final accuracy
    accuracy = total_correct / total_samples
    logger.info(f'Testing complete. Final accuracy: {accuracy:.4f}')

    return accuracy

