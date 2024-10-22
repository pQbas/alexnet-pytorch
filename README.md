<div align="center">
  <h1>alexnet-pytorch</h1>
  AlexNet CNN implemented in PyTorch, ready for training, testing, and inference.
  <!--  -->
  <!-- <p align="center"> -->
  <!--   <a href="here_is_a_demo_video"> -->
  <!--   <img alt="Blueberry Detection ROS" src="gallery/image-demo.png"></a> -->
  </p>

</div>


## How install?

Go to the folder of this project and run pip install.
```
cd alexnet-pytorch
pip install .
```

## How use it?

This library gives access for three main actions with the alexnet-cnn, this actions are
`train`, `test` and `inference`. The `demo` folder contains an example of how use it
with a notebook ready to use in colab. Below are some snippets wich explains the code 
in the demo folder.


### Train action

Following code helps you to train alexnet. To train is needed to define a `CONFIG_PARAMS`
constant, this is a dictionary that contains training parameters such as `batch size`,
`categories`, `optimizer`, `learning rate`, etc. The `train` function receives this
dictionary and gives you the path where the weights were saved as a `pt` file.

```python
# Import alexnet library previously installed
import alexnet

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
```

### Test action

Result of this action is the accuracy metric computed for the trained model, this
function receives the `paras` paramtere and also the `weights path`.

```python
# Import alexnet library previously installed
import alexnet

# Test the AlexNet model
accuracy = alexnet.test(params      = CONFIG_PARAMS, 
                        weightsPath = weightsPath)
```

### Inference action

Inference receives an image, model and the device as input, and gives you the category
of the image. In following example is used PIL to load the image, and some utilities
as for loading the model and getting the device. 

```python
# Import alexnet library previously installed
import alexnet
from PIL import Image

# Constat with an image to perform the testing
IMG_PATH = '../gallery/cat.jpeg'

# Getting the main device to perform inference `gpu` by defult.
DEVICE = alexnet.utils.getDevice()

# Load model the trained model and image 
model = alexnet.utils.loadModel(weightsPath = weightsPath, 
                                params      = CONFIG_PARAMS, 
                                device      = DEVICE)
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


