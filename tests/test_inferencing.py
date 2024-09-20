import pytest
from PIL import Image
from alexnet import inference


def test_testing_preprocess():
    IMG_PATH = '/test/gallery/cat.jpeg'

    img = Image.open(IMG_PATH)

    preprocessImage = inference.preprocess(img)

    print(preprocessImage)
    
