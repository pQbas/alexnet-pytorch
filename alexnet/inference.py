from alexnet.model import AlexNet
from alexnet.utils import getConfig, INFERENCE_TRANSFORM
import torch.nn as nn
import torch

import logging
logger = logging.getLogger(__name__)  # Get logger for this module


def preprocess(inputData):
    '''
    Apply preprocessing to the Data.
    inputData is something like an OpenCV image.
    Output of this is a tensor image normalized.
    '''
    if not inputData:
        logger.error("InputData is not a valid type")
        raise ValueError("inputData is not a valid type") 

    logger.debug("Preprocessing input data")
    preprocessData = INFERENCE_TRANSFORM(inputData)  # Assuming INFERENCE_TRANSFORM is defined elsewhere
    return preprocessData

def predict(model, preprocessedData):
    '''
    Perform prediction over the preprocessedData.
    '''
    logger.debug("Performing forward pass on preprocessed data")
    output = model(preprocessedData)
    probs = nn.Softmax(dim=1)(output)
    _, indices = torch.max(probs, dim=1)
    
    prediction = indices.item() if torch.is_tensor(indices) else None
    logger.debug(f'Raw prediction result: {prediction}')
    
    return prediction 

def postprocess(prediction):
    '''
    Apply post-processing to make the probabilities and convert
    it into a more interpretable form.
    '''
    logger.debug("Postprocessing the prediction result")
    return postProcessPrediction  # Assuming postProcessPrediction is defined elsewhere

def inference(input, model, device):
    '''
    Main function to handle inference.
    - Preprocesses input
    - Predicts using the model
    - (Optionally) Postprocesses the prediction
    '''
    logger.info('\n========== INFERENCE ==========\n')
    logger.info(f'Using device: {device}')

    # Preprocess the input data
    logger.info('Step 1: Preprocessing input data')
    preprocessData = preprocess(input)
    logger.debug(f'Preprocessed data shape: {preprocessData.shape}')

    # Perform prediction
    logger.info('Step 2: Performing prediction')
    prediction = predict(model.to(device), preprocessData[None, ...].to(device))
    logger.info(f'Prediction result: {prediction}')

    # (Optional) Postprocess the prediction
    # logger.info('Step 3: Postprocessing the prediction (if needed)')
    # postprocessData = postprocess(prediction)

    return prediction

