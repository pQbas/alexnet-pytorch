# alexnet-pytorch
Alexnet cnn implemented in pytorch


## How install?

Go to the folder of this project and run pip install.
```
cd alexnet-pytorch
pip install .
```

## How use it?

This library gives access for three main actions with the alexnet-cnn, this actions are
`train`, `test` and `inference`. The `demo` folder contains an example of how use it. Here
is a snippet with the code in the demo folder.

```
# Import alexnet library previously installed
import alexnet
from PIL import Image

# Constat with an image to perform the testing
IMG_PATH = '../gallery/cat.jpeg'

# Getting the main device to perform inference `gpu` by defult.
DEVICE = alexnet.utils.getDevice()

# Define the config params for all proceess
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
weightsPath = alexnet.train(params = CONFIG_PARAMS)

# Test the AlexNet model
accuracy = alexnet.test(params = CONFIG_PARAMS, weightsPath=weightsPath)

# Load model the trained model and image 
model = alexnet.utils.loadModel(weightsPath = weightsPath, params = CONFIG_PARAMS, device=DEVICE)
image = Image.open(IMG_PATH)

# Perform inference (preprocessing and prediction)
results = alexnet.inference(image, model, DEVICE)
```

### Using with Docker

Run following command at `alexnet` directory, this command creates a container with a share 
volume with the files of the directory. After run this change to the `workspace` folder, 
and follow the steps to install as usual.

```bash
docker build -t alexnet-container .
docker run -it -v $PWD:/workspace --gpus all alexnet-container
```


