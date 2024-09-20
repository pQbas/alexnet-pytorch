import os

import torch
import torch.optim as optim
import torch.nn as nn

import numpy as np
from tqdm import tqdm
from datetime import datetime

from alexnet.model import AlexNet
# import alexnet.utils as utils
from alexnet.utils import getConfig, getDataset, getDevice, buildDataloader, loadModel
# getConfig, getDataset, getDevice, buildDataloader


def saveModel(
    model : nn.Module,
    name  : str,
    path  : str
    ):
    '''
    Saves model at directory defined by path with 
    a given name, return true if the model was 
    saved succesfully and false if not
    '''

    current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")
    else:
        print(f"Folder '{path}' already exists.")

    modelName = f'{name}-{current_time}.pt'  
    
    torch.save(model.state_dict(), os.path.join(path, modelName))
    print(f"Weights '{modelName}' were saved at '{path}'")

    return


def buildLoss(
    typeLoss : str
):
    '''
    Return the loss function given by the
    typeLoss parameter
    '''

    loss = None

    if typeLoss == 'binary-cross-entropy':
        loss = nn.BCELoss() 

    if typeLoss == 'cross-entropy':
        loss = nn.CrossEntropyLoss()

    return loss


def buildOptimizer(
    model : nn.Module,
    typeOptimizer : str,
    learningRate  : float
):
    '''
    Return the optimizer function given by the
    typeOptimizer parameter
    '''
    optimizer = None

    if typeOptimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=learningRate)
    elif typeOptimizer == 'adadelta':
        optimizer = optim.Adadelta(model.parameters(), lr=learningRate)
    else:
        raise ValueError("At buildOptimizer(), typeOptimizer must be one of the valid options")

    return optimizer


def buildModel(
    numCategories : int,
    device        : str
):
    dummyInput = torch.rand([1, 3, 224, 224]).to(device)
    model = AlexNet(categories = numCategories).to(device)
    model(dummyInput)
    return model



def testEpoch(
    model,
    testLoader,
    device
):
    '''
    Compute the accuracy for the model with the
    testLoader
    '''
    total_correct, total_samples = 0, 0

    threshold = 0.5
    model.eval() 
     
    pbar = tqdm(testLoader, unit='batch', desc='description')
    
    for data in pbar:
        pbar.set_description('Testing progress')

        inputs, labels = data[0].to(device), data[1].to(device)

        outputs = model(inputs)
        probs  = nn.Softmax(dim=1)(outputs)

        values, indices = torch.max(probs, dim=1)

        total_correct += torch.sum((indices == labels) * (values >= threshold)).item()
        total_samples += labels.size(0) 
    
    accuracy = total_correct / total_samples
    return accuracy


def trainEpoch(
    model,
    trainLoader,
    optimizer,
    lossf,
    device
):
    '''
    Perform training of the model with the train-set,
    optimizer, and loss function, returns the model trained
    '''

    model.train()
    runningLoss = []

    pbar = tqdm(trainLoader, unit='batch', desc='description')

    for data in pbar:
        pbar.set_description('Training progress')

        inputs, labels = data[0].to(device), data[1].to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        probs  = nn.Softmax(dim=1)(outputs)
        
        loss = lossf(probs, labels)
        loss.backward()
        optimizer.step()

        runningLoss.append(loss.item())

    return np.mean(runningLoss)


def train(
    paramsPath : str
):
    params = getConfig(paramsPath)

    device = getDevice()

    trainset, testset = getDataset(name = params['dataset_name'])

    trainloader = buildDataloader(trainset,
                                  batchsize  = int(params['batch_size']))

    testloader = buildDataloader(testset,
                                 batchsize = int(params['batch_size']))

    model = buildModel(numCategories = int(params['categories']),
                       device        = device)
    
    optimizer = buildOptimizer(model,
                               typeOptimizer = params['optimizer'],
                               learningRate  = params['learning_rate']) 
    
    lossf = buildLoss(typeLoss = params['loss'])
    
    for i in range(int(params['epochs'])):
        print(f" > Epoch {i}/{params['epochs']}")

        loss = trainEpoch(model, trainloader, optimizer, lossf, device) 
        acc  = testEpoch(model, testloader, device)
        
        print('Training Loss:', loss, '|', 'Testing Accuracy:', acc)

    saveModel(model, 
              name = params['name'], 
              path = params['path'])
    
    torch.cuda.empty_cache()
    return


