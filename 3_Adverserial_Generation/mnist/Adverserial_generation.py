from sklearn.model_selection import train_test_split
import pandas as pd

import numpy as np
import keras
from keras import backend
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf

from cleverhans.attacks import FastGradientMethod, BasicIterativeMethod, SaliencyMapMethod, CarliniWagnerL2
from cleverhans.utils_keras import KerasModelWrapper

from matplotlib import pyplot as plt
import imageio

from scipy.misc import imsave
import os

def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x.reshape(x.shape[1], x.shape[2])  # original shape (1,img_rows, img_cols,1)

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

np.set_printoptions(threshold=np.nan)
# Set the matplotlib figure size
plt.rc('figure', figsize = (12.0, 12.0))

# Set the learning phase to false, the model is pre-trained.
backend.set_learning_phase(False)
keras_model = load_model("LeNet-1_200_Epochs.h5")

tf.set_random_seed(1234)

if not hasattr(backend, "tf"):
    raise RuntimeError("This tutorial requires keras to be configured"
                       " to use the TensorFlow backend.")

if keras.backend.image_dim_ordering() != 'tf':
    keras.backend.set_image_dim_ordering('tf')
    print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to "
          "'th', temporarily setting to 'tf'")

# Retrieve the tensorflow session
sess =  backend.get_session()

x = tf.placeholder(tf.float32, shape=(None, 28, 28, 1))
y = tf.placeholder(tf.float32, shape=(None, 10))

#i believe x_test contains like 10k images
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_test = x_test.astype('float32')
x_test /= 255

x_test = x_test.reshape(x_test.shape[0], 28,28,1)
#y_test = np_utils.to_categorical(y_test, 10)

x_validation = x_test[::100]
y_validation = y_test[::100] #this gets like 100 validation images

path = "base_set"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(x_validation):
    imsave(path + '/' +str(idx)+'.png',x.reshape(28,28))

pred = np.argmax(keras_model.predict(x_validation),axis=1)
acc =  np.mean(np.equal(pred, y_validation))

print("The normal validation accuracy is: {}".format(acc))

# Initialize the Fast Gradient Sign Method (FGSM) attack object and
# use it to create adversarial examples as numpy arrays.
wrap = KerasModelWrapper(keras_model)
fgsm = FastGradientMethod(wrap, sess=sess)
fgsm_params = {'eps': 0.1,
               'clip_min': 0.,
               'clip_max': 1.}


adv_x = fgsm.generate_np(x_validation, **fgsm_params)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

path = "leNet1/fgsm"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(adv_x):
    imsave(path + '/' +str(idx)+'.png',x.reshape(28,28))

print("The adversarial validation accuracy is: {}".format(adv_acc))

x_sample = x_validation[10].reshape(28, 28)
adv_x_sample = adv_x[10].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()

bim = BasicIterativeMethod(wrap, sess=sess)


adv_x = bim.generate_np(x_validation)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

path = "leNet1/bim"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(adv_x):
    imsave(path + '/' +str(idx)+'.png',x.reshape(28,28))

print("The adversarial validation accuracy is: {}".format(adv_acc))

x_sample = x_validation[10].reshape(28, 28)
adv_x_sample = adv_x[10].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()


smm = SaliencyMapMethod(wrap, sess=sess)
smm_params = {'theta': 0.1,
                'gamma':0.1,
               'clip_min': 0.,
               'clip_max': 1.}


adv_x = smm.generate_np(x_validation, **smm_params)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

path = "leNet1/smm"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(adv_x):
    imsave(path + '/' +str(idx)+'.png',x.reshape(28,28))

print("The adversarial validation accuracy is: {}".format(adv_acc))

x_sample = x_validation[10].reshape(28, 28)
adv_x_sample = adv_x[10].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()


cwl2 = CarliniWagnerL2(wrap, sess=sess)

adv_x = cwl2.generate_np(x_validation)

adv_pred = np.argmax(keras_model.predict(adv_x), axis = 1)
adv_acc =  np.mean(np.equal(adv_pred, y_validation))

path = "leNet1/cwl2"
try:
    os.makedirs(path)
except OSError:
    print ("Creation of the directory %s failed" % path)
else:
    print ("Successfully created the directory %s" % path)

for idx, x in enumerate(adv_x):
    imsave(path + '/' +str(idx)+'.png',x.reshape(28,28))

print("The adversarial validation accuracy is: {}".format(adv_acc))

x_sample = x_validation[10].reshape(28, 28)
adv_x_sample = adv_x[10].reshape(28, 28)

adv_comparison = stitch_images([x_sample, adv_x_sample], 1, 2)

plt.imshow(adv_comparison)
plt.show()
