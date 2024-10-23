import torch
from PIL import Image
import alexnet

# Constants for file paths
CONFIG_FILE_PATH = 'config.ini'
WEIGHTS_PATH = 'runs/alexnet.pt'
IMG_PATH = '../gallery/cat.jpeg'

# Device configuration using utils
DEVICE = alexnet.utils.getDevice()


CONFIG_PARAMS = {
    'batch_size'    : 16,
    'categories'    : 10,
    'optimizer'     : 'sgd',
    'learning_rate' : 0.001,
    'loss'          : 'cross-entropy',
    'epochs'        : 5,
    'model_name'    : 'alexnet',
    'path'          : 'runs',
    'dataset_name'  : 'cifar10',
}

# Train the AlexNet model
weightsPath = alexnet.train(params = CONFIG_PARAMS,
                            tracking_train = True)

# Test the AlexNet model
accuracy = alexnet.test(params = CONFIG_PARAMS, weightsPath=weightsPath)

# Load model and perform inference
model = alexnet.utils.loadModel(weightsPath = weightsPath, params = CONFIG_PARAMS, device=DEVICE)
image = Image.open(IMG_PATH)

# Perform inference (preprocessing and prediction)
results = alexnet.inference(image, model, DEVICE)


