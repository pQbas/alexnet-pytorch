import torch
from PIL import Image
import alexnet

# Constants for file paths
CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/runs/alexnet.pt'
IMG_PATH = '/home/pqbas/projects/alexnet-pytorch/gallery/cat.jpeg'

# Device configuration using utils
DEVICE = alexnet.utils.getDevice()

# Train the AlexNet model
alexnet.train(paramsPath=CONFIG_FILE_PATH)

# Test the AlexNet model
accuracy = alexnet.test(paramsPath=CONFIG_FILE_PATH, weightsPath=WEIGHTS_PATH)

# Load model and perform inference
model = alexnet.utils.loadModel(WEIGHTS_PATH, CONFIG_FILE_PATH, device=DEVICE)
image = Image.open(IMG_PATH)

# Perform inference (preprocessing and prediction)
results = alexnet.inference(image, model, DEVICE)


