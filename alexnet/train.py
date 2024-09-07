import argparse
import wandb
import yaml
import torch
import torch.optim as optim
import torch.nn as nn
import os

def saveModel(
    model,
    name : str,
    path : str
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
    typeloss : str
):
    '''
    Return the loss function given by the
    typeLoss parameter
    '''

    loss = None

    if typeloss == 'bce':
        loss = nn.BCELoss() 

    return loss


def buildOptimizer(
    model,
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
    numCategories : int
):
    dummyInput = torch.rand([1, 1, 224, 224]).to(device)
    model = AlexNet(numCategories = numCategories).to(device)
    model(dummyInput)
    return model




def buildDataloader(
    dataset,
    batchsize  : int,
    typeloader : str
):
    '''
    Build a dataloader to the dataset with a batch
    -size and a typeLoader (train, test, val), for
    different use
    '''
    return


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
    loss
):
    '''
    Perform training of the model with the train-set,
    optimizer, and loss function, returns the model trained
    '''

    return

def getDataset(
    path : str
):
    '''
    Returns a dataset
    '''
    return

def getConfig(
    path : str
) -> dict:
    config = None

    with open(path, 'r') as stream:
        config = yaml.safe_load(stream)

    if not (config):
        print('Verify your the file path in argument')
        exit()

    return config

def run(
    params : dict
):

    dataset     = getDataset(path = params['dataset_path'])

    trainloader = buildDataloader(dataset,
                                  batchsize  = params['batch_size'],
                                  typeloader = 'train')

    testloader = buildDataloader(dataset,
                                 batchsize  = params['batch_size'],
                                 typeloader = 'test')

    model     = buildModel(numCategories = params['categories'])

    optimizer = buildOptimizer(model,
                               typeOptimizer = params['optimizer'],
                               learningRate  = params['learning_rate']) 

    lossf = buildLoss(typeloss = params['loss'])

    for _ in range(params['config']):
        loss = trainEpoch(model, trainloader, optimizer, lossf)
        acc  = testEpoch(model, testloader)

    saveModel(model, 
              name = params['name'], 
              path = params['path'])

    torch.cuda.empty_cache()

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--config', 
                        type=str, 
                        default='config.yaml', 
                        help='Configuration file for training')

    args = parser.parse_args()

    config = getConfig(args.config)
    run(params = config['parameters'])
