from alexnet.model import AlexNet
from alexnet.utils import getConfig, INFERENCE_TRANSFORM
import torch.nn as nn
import torch


def preprocess(
    inputData
    ):
    '''
    Apply preproceesing to the Data

    inputData is something like an OpenCV image,
    output of this is a tensor image normalized
    '''
    if not(inputData): raise ValueError("inputData is not a valid type") 
    preprocessData = INFERENCE_TRANSFORM(inputData)
    return preprocessData


def predict(model, preprocessedData):
    '''
    Perform prediction over the preprocessedData
    '''
    output = model(preprocessedData)
    probs = nn.Softmax(dim=1)(output)
    values, indices = torch.max(probs, dim=1)
    prediction = indices if torch.is_tensor(indices) else None
    return prediction 


def postprocess(prediction):
    '''
    Apply post processing over prediction to make
    the probabilities and convert it into a json
    file with ineterpretable predictions
    '''

    return postProcessPrediction



def inference(input, weightsPath):

    model = loadModel(weightsPath)

    preprocessData = preprocess(input)

    prediction = predict(preprocessData)

    postprocessData = postprocess(prediction)

    return postprocessData
