# alexnet-pytorch
Alexnet cnn implemented in pytorch


### Steps to work

Run following command at `alexnet` directory, this command creates a container with a share volume with the files of the directory. After run this change to the `test` folder, and work as usual.

'''
docker run -it -v $PWD:/test --gpus all pytorch/pytorch
cd ../test
'''


