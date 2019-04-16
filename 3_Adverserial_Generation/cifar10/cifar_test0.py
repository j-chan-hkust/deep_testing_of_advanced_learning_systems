from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod, SaliencyMapMethod, CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import imageio

from vgg16_CIFAR10 import cifar10vgg
import os
from scipy.misc import imsave



np.set_printoptions(threshold=np.nan)
# Set the matplotlib figure size
plt.rc('figure', figsize = (12.0, 12.0))

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)
vgg = cifar10vgg(train=False)
keras_model = vgg.model

tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess = backend.get_session()

#i believe x_test contains like 10k images
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_test = x_test.astype('float32')
x_train, x_test = vgg.normalize(x_train, x_test)

x_test = x_test.reshape(x_test.shape[0], 32,32,3)
#y_test = np_utils.to_categorical(y_test, 10)

x_validation = x_test[::100]
y_validation = y_test[::100].flatten() #this gets like 100 validation images

pred = np.argmax(keras_model.predict(x_validation),axis=1)
acc =  np.mean(np.equal(pred, y_validation))

path = "cifar10/base_set"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(x_validation):
    imsave(path + '/' +str(idx)+'.png',x.reshape(32,32,3))

print("The normal validation accuracy is: {}".format(acc))

# Initialize the Fast Gradient Sign  Method (FGSM) attack object and
# use it to create adversarial examples as numpy arrays.
wrap = KerasModelWrapper(keras_model)
bim = SaliencyMapMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.1,
               'clip_min': 0.,
               'clip_max': 1.}


adv_x = bim.generate_np(x_validation)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

print("The adversarial validation accuracy is: {}".format(adv_acc))

def stitch_images(images, y_img_count, x_img_count, margin = 2):

    # Dimensions of the images
    img_width = images[0].shape[0]
    img_height = images[0].shape[1]

    width = y_img_count * img_width + (y_img_count - 1) * margin
    height = x_img_count * img_height + (x_img_count - 1) * margin
    stitched_images = np.zeros((width, height, 3))

    # Fill the picture with our saved filters
    for i in range(y_img_count):
        for j in range(x_img_count):
            img = images[i * x_img_count + j]
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            stitched_images[(img_width + margin) * i: (img_width + margin) * i + img_width,
                            (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    return stitched_images

x_sample = x_validation[1].reshape(32, 32,3)
adv_x_sample = adv_x[1].reshape(32, 32,3)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()
