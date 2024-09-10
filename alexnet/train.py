import argparse
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import os
from model import AlexNet
import torchvision
import torchvision.transforms as transforms
from config import Config
import numpy as np
from tqdm import tqdm, trange


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
    
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Folder '{path}' created successfully.")
    else:
        print(f"Folder '{path}' already exists.")
    
    torch.save(model.state_dict(), os.path.join(path, f'{name}.pt'))

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

    return optimizer


def buildModel(
    numCategories : int,
    device        : str
):
    dummyInput = torch.rand([1, 3, 224, 224]).to(device)
    model = AlexNet(categories = numCategories).to(device)
    model(dummyInput)
    return model


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
                                         sampler=np.random.permutation(20))
    return loader


def testEpoch(
    model,
    testLoader
):
    '''
    Compute the accuracy for the model with the
    testLoader
    '''

    return metrics


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


def getDataset(
    name : str,
    path : str = None
):
    '''
    Returns a dataset
    '''
    trainset, testset = None, None
    
    if name == 'cifar':
        transform = transforms.Compose(
                [transforms.Resize((224,224)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        trainset = torchvision.datasets.CIFAR10(root='./data', 
                                                train=True,
                                                download=True, 
                                                transform=transform)
        
        testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform)

    return (trainset, testset)


def getDevice() -> str:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return device


def run(
    params : dict
):
    device = getDevice()

    trainset, testset = getDataset(name = 'cifar')

    trainloader = buildDataloader(trainset,
                                  batchsize  = params['batch_size'])

    testloader = buildDataloader(testset,
                                 batchsize = params['batch_size'])

    model = buildModel(numCategories = params['categories'],
                       device        = device)
    
    optimizer = buildOptimizer(model,
                               typeOptimizer = params['optimizer'],
                               learningRate  = params['learning_rate']) 
    
    lossf = buildLoss(typeLoss = params['loss'])
    
    for i in range(params['epochs']):

        print(f"Epoch {i}/{params['epochs']}")

        loss = trainEpoch(model, trainloader, optimizer, lossf, device)
        print('Training loss:', loss)
        
        acc  = testEpoch(model, testloader)
    
    # saveModel(model, 
    #           name = params['name'], 
    #           path = params['path'])
    #
    # torch.cuda.empty_cache()

    return


if __name__ == '__main__':

    # parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--config', 
    #                     type=str, 
    #                     default='config.yaml', 
    #                     help='Configuration file for training')
    #
    # args = parser.parse_args()
    #
    # config = getConfig(args.config)


    config = Config()
    run(params = config.params)


