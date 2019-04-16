from keras.applications.vgg16 import VGG16
from keras.layers import Input, Dense
from keras.models import load_model, Model, Sequential
from keras.datasets import cifar10
from keras.utils import np_utils
from keras import models, layers, activations
import keras
import argparse
import numpy as np
import sys
from vgg16_CIFAR10 import cifar10vgg

# input image dimensions
img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

VGG_model = cifar10vgg(train=False)

VGG_model.model.summary()

#define the layer depth here
layer_depth = -6
model_path = ""

wgts = VGG_model.model.layers[layer_depth].get_weights()
nthLayerNeurons = VGG_model.model.layers[layer_depth].output_shape[1]

last_layer = VGG_model.model.layers[layer_depth-1].output

new_model = Dense(nthLayerNeurons, activation=None, name='nthlayer_no_activation')(last_layer)
new_model = Model(VGG_model.model.input, new_model, name='nthlayer_no_activation')

new_model.summary()
# with the new layer, load the previous weights
new_model.layers[-1].set_weights(wgts)

new_model.summary()

#now that the model is initialized, we do load cifar10 dataset.
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
#because we dont use the original model, we must normalize
x_train, x_test = VGG_model.normalize(x_train, x_test)

intermediate_output = new_model.predict(x_train)


print('output of nth layer before the activation function:')
print(intermediate_output)

print('model1 has been saved into the file: ' + "VGG16" + "_"+ str(layer_depth) + "th_layer_data.npy")
np.save('VGG16' + "_"+ str(layer_depth) + "th_layer_data" ,np.array(intermediate_output))
