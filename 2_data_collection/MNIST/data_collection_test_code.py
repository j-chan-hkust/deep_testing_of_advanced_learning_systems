# this file is meant to collect data of neuron output for the training set
# this only applies to the MNIST dataset
# will work on a generalized version later

from keras.models import load_model, Model
from keras.datasets import mnist
from keras.utils import np_utils
from keras import models, layers, activations
import keras
import argparse
import numpy as np


#function from: https://stackoverflow.com/questions/49492255/how-to-replace-or-insert-intermediate-layer-in-keras-model
def replace_intermediate_layer_in_keras(model, layer_id, new_layer):

    layers = [l for l in model.layers]

    x = layers[0].output
    for i in range(1, len(layers)):
        if i == layer_id:
            x = new_layer(x)
        else:
            x = layers[i](x)

    new_model = Model(input=layers[0].input, output=x)
    return new_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Data Collector for neuron outputs')
    #parser.add_argument('dataset', type=str, help='what dataset is this model for?')
    #parser.add_argument('model_path', type=str, help='which model?')
    args = parser.parse_args()

    # get the mnist dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train /= 255
    x_test /= 255
    y_train = np_utils.to_categorical(y_train, 10)
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train.reshape(x_train.shape[0], 28,28,1)
    x_test = x_test.reshape(x_test.shape[0], 28,28,1)

    model = keras.models.load_model("LeNet-5_200_Epochs.h5")#args.model_path)
    updatedModel = keras.models.load_model("LeNet-5_200_Epochs.h5")
    # we get the neuron output for the penultimate layer for each neuron

    # taken from the suggestion at: https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
    # we recreate the model, inserting a new layer that does nothing BEFORE the layer we want to calculate.

    #make a new model

    nthLayer = model.layers[6].output

    print(type(model.layers[6].output_shape))

    #remove the last layer and penultimate layer
    updatedModel.pop()
    updatedModel.pop()

    # add new layer with no activation
    updatedModel.add(layers.Dense(model.layers[6].output_shape[1],activation = None))

    # get the weights from original, copy to new
    wgts = model.layers[6].get_weights()
    updatedModel.layers[6].set_weights(wgts)


    intermediate_layer_model = Model(inputs=updatedModel.input,
                             outputs=updatedModel.layers[6].output )

    intermediate_output = intermediate_layer_model.predict(x_train)

    compareintermediate_layer_model = Model(inputs=model.input,
                             outputs=model.layers[6].output )

    compareintermediate_output = compareintermediate_layer_model.predict(x_train)

    updatedModel.summary()
    model.summary()

    print(intermediate_output)
    print(compareintermediate_output)
    #np.save("LeNet-5_data.npy",np.array(intermediate_output))
