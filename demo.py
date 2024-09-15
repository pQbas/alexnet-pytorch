import cv2
from alexnet.train import train, getConfig
# from alexnet.inference import inference  
from alexnet.test import test 

# Training in a given dataset

CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
WEIGHTS_PATH = '/home/pqbas/projects/alexnet-pytorch/runs/alexnet.pt'


train(paramsPath = CONFIG_FILE_PATH)

accuracy = test(paramsPath = CONFIG_FILE_PATH, weightsPath = WEIGHTS_PATH)

print(accuracy) 

# Training in a given dataset
#
# CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'
#
# config = getConfig(CONFIG_FILE_PATH)
#
# train(params = config)
#
# Inference over an image


#
# IMG_FILE_PATH = ''
# WEIGHTS_PATH = ''
#
#
# img = cv2.imread(IMG_FILE_PATH)
#
# result = inference(input = img)

# Validation over a dataset

# DATASET_PATH = ''
#
# metrics = test(params = config)
#

# ??? How make transfer learning ... with this?
# ??? How use alexnet as an object i mean
#     -  alexnet.train ...?
#     -  alexnet.inference ...?
#
