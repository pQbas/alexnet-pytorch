from alexnet.model import AlexNet

def loadModel(weightsPath):
    '''
    Loads the weights path to the model, and
    return the model
    '''

    model = load...

    return model


def preprocess(inputData):
    '''
    Apply preproceesing to the Data
    '''

    return preprocessData



def postprocess(prediction):
    '''
    Apply post processing over prediction to make
    the probabilities and convert it into a json
    file with ineterpretable predictions
    '''

    return postProcessPrediction


def predict(preprocessedData):
    '''
    Perform prediction over the preprocessedData
    '''

    return prediction


def inference(input, weightsPath):

    model = loadModel(weightsPath)

    preprocessData = preprocess(input)

    prediction = predict(preprocessData)

    postprocessData = postprocess(prediction)

    return postprocessData
