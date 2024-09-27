import torch
from PIL import Image
import alexnet

# Constants for file paths
CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/runs/alexnet.pt'
IMG_PATH = '/home/pqbas/projects/alexnet-pytorch/gallery/cat.jpeg'

# Device configuration using utils
DEVICE = alexnet.utils.getDevice()


CONFIG_PARAMS = {
    'batch_size' : 16,
    'categories' : 10,
    'optimizer' : 'sgd',
    'learning_rate' : 0.001,
    'loss' : 'cross-entropy',
    'epochs' : 5,
    'model_name' : 'alexnet',
    'path' : 'runs',
    'dataset_name' : 'cifar10',
}

# paramsPath=CONFIG_FILE_PATH
# paramsPath=CONFIG_FILE_PATH,

# Train the AlexNet model
alexnet.train(params = CONFIG_PARAMS)

# Test the AlexNet model
accuracy = alexnet.test(params = CONFIG_PARAMS, weightsPath=WEIGHTS_PATH)

# Load model and perform inference
model = alexnet.utils.loadModel(weightsPath = WEIGHTS_PATH, params = CONFIG_PARAMS, device=DEVICE)
image = Image.open(IMG_PATH)

# Perform inference (preprocessing and prediction)
results = alexnet.inference(image, model, DEVICE)


