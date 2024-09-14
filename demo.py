from alexnet.train import train, getConfig

# Training in a given dataset

CONFIG_FILE_PATH = '/home/pqbas/projects/alexnet-pytorch/alexnet/config.ini'

config = getConfig(CONFIG_FILE_PATH)

train(params = config)
