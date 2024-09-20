import pytest
from PIL import Image
from alexnet import inference
from alexnet import utils
import torch

def test_testing_preprocess():
    IMG_PATH = '/test/gallery/cat.jpeg'
    img = Image.open(IMG_PATH)
    preprocessImage = inference.preprocess(img)
    return

def test_testing_prediction():
    WEIGHTS_PATH = '/test/runs/alexnet.pt'
    PARAMS_PATH = '/test/alexnet/config.ini'
    DEVICE = utils.getDevice()

    model = utils.loadModel(WEIGHTS_PATH, PARAMS_PATH, device = DEVICE)
    preprocessImage = torch.rand([4, 3, 224, 224]).to(DEVICE)
    results = inference.predict(model, preprocessImage)
    return
