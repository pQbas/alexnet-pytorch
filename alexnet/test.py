from alexnet.model import AlexNet
from alexnet.train import getConfig, getDataset, getDevice, buildDataloader
import torch
import torch.nn as nn
from tqdm import tqdm

CONFIG_PARAMS = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/runs/alexnet.pt'


def test(
    paramsPath  : str, 
    weightsPath : str
):
    # params config and device
    params = getConfig(paramsPath)
    device = getDevice()

    # data to test
    _, testset = getDataset(name = params['dataset_name'])

    testLoader = buildDataloader(testset,
                                 batchsize = int(params['batch_size']))

    # load model
    model = AlexNet(categories = int(params['categories']))
    model.load_state_dict(torch.load(weightsPath, map_location=device))
    model.to(device)
    model.eval()

    # initial values of metrics
    total_correct, total_samples = 0, 0

    # start veluation
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
