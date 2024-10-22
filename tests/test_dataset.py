import pytest
from unittest.mock import patch
import torchvision
from alexnet.utils import getDataset  # Replace 'your_module' with the actual module name

# Mocking the CIFAR10 and MNIST datasets
class MockCIFAR10:
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

class MockMNIST:
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

class MockFashionMNIST:
    def __init__(self, root, train, download, transform):
        self.root = root
        self.train = train
        self.download = download
        self.transform = transform

# Test for CIFAR10 dataset
@patch('torchvision.datasets.CIFAR10', side_effect=MockCIFAR10)
def test_getDataset_cifar10(mock_cifar10):
    trainset, testset = getDataset('cifar10', './mock_path')
    
    # Assert that the datasets are correctly initialized
    assert trainset is not None
    assert testset is not None
    assert isinstance(trainset, MockCIFAR10)
    assert isinstance(testset, MockCIFAR10)
    assert trainset.train is True
    assert testset.train is False

# Test for MNIST dataset
@patch('torchvision.datasets.MNIST', side_effect=MockMNIST)
def test_getDataset_mnist(mock_mnist):
    trainset, testset = getDataset('mnist', './mock_path')
    
    # Assert that the datasets are correctly initialized
    assert trainset is not None
    assert testset is not None
    assert isinstance(trainset, MockMNIST)
    assert isinstance(testset, MockMNIST)
    assert trainset.train is True
    assert testset.train is False

# Test for FashionMNIST dataset
@patch('torchvision.datasets.FashionMNIST', side_effect=MockFashionMNIST)
def test_getDataset_fashionmnist(mock_fashionmnist):
    trainset, testset = getDataset('fashion_mnist', './mock_path')
    
    # Assert that the datasets are correctly initialized
    assert trainset is not None
    assert testset is not None
    assert isinstance(trainset, MockFashionMNIST)
    assert isinstance(testset, MockFashionMNIST)
    assert trainset.train is True
    assert testset.train is False


# Test for invalid dataset name
def test_getDataset_invalid_name():
    with pytest.raises(ValueError, match="At getDataset 'name' param must be one of valid datasets"):
        getDataset('invalid_dataset')

