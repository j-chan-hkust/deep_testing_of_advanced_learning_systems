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
import sys

# TODO: doesnt really scale to working with layers that arent Dense. need to account for that

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
    parser.add_argument('dataset', type=str, help='what dataset is this model for?')
    parser.add_argument('model_path', type=str, help='which model?')
    parser.add_argument('layer_depth', type=int, help='which layer? (only positive values)')
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

    model = keras.models.load_model(args.model_path)
    # we get the neuron output for the penultimate layer for each neuron

    # implemented with help from the suggestion at: https://stackoverflow.com/questions/45492318/keras-retrieve-value-of-node-before-activation-function
    # we recreate the model, delete layers up to and including the layer we want to analyze, add a blank layer with no activation, and then import the old weights to this layer.

    #make a new model

    # some simple input checks
    if(args.layer_depth < 0):
        println ('layer depth must be positive!')
        sys.exit()

    if(args.layer_depth > len(model.layers)):
        println ('layer depth too large!')
        sys.exit()

    # save the original weights
    wgts = model.layers[args.layer_depth].get_weights()
    nthLayerNeurons = model.layers[args.layer_depth].output_shape[1]

    #remove layers up to the nth layer
    for i in range(len(model.layers)-args.layer_depth):
        model.pop()
    model.summary()
    # add new layer with no activation
    model.add(layers.Dense(nthLayerNeurons,activation = None))

    # with the new layer, load the previous weights
    model.layers[args.layer_depth].set_weights(wgts)

    # get the output of this new model.
    intermediate_layer_model = Model(inputs=model.input,
                             outputs=model.layers[args.layer_depth].output )

    intermediate_output = intermediate_layer_model.predict(x_train)


    print('output of nth layer before the activation function:')
    print(intermediate_output)

    print('model has been saved into the file: ' + str(args.model_path)[0:-3] + "_"+ str(args.layer_depth) + "th_layer_data.npy")
    np.save(str(args.model_path)[0:-3] + "_"+ str(args.layer_depth) + "th_layer_data" ,np.array(intermediate_output))
