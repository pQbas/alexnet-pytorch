import pytest
import torch
from PIL import Image
import alexnet

def test_testing_inference():
    CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
    WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/runs/alexnet.pt'
    IMG_PATH = '/home/pqbas/projects/alexnet-pytorch/gallery/cat.jpeg'

    # Device configuration using utils
    DEVICE = alexnet.utils.getDevice()

    # Load model and perform inference
    model = alexnet.utils.loadModel(weightsPath=WEIGHTS_PATH, paramsPath=CONFIG_FILE_PATH, device=DEVICE)
    image = Image.open(IMG_PATH)

    # Perform inference (preprocessing and prediction)
    results = alexnet.inference(image, model, DEVICE)

    print('Results:', results)
