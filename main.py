#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""This showcases how simple it is to build image classification networks.

It follows description from this TensorFlow tutorial:
    https://www.tensorflow.org/versions/master/tutorials/mnist/pros/index.html#deep-mnist-for-experts
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from time import time

from sklearn import metrics
import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.learn as learn
import matplotlib.pyplot as plt
import colorama
from colorama import Fore, Back, Style




def max_pool_2x2(tensor_in):
  return tf.nn.max_pool(
      tensor_in, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def conv_model(feature, target, mode):
  """2-layer convolution model."""
  # Convert the target to a one-hot tensor of shape (batch_size, 10) and
  # with a on-value of 1 for each one-hot vector of length 10.
  target = tf.one_hot(tf.cast(target, tf.int32), 10, 1, 0)

  # Reshape feature to 4d tensor with 2nd and 3rd dimensions being
  # image width and height final dimension being the number of color channels.
  feature = tf.reshape(feature, [-1, 28, 28, 1])

  # First conv layer will compute 32 features for each 5x5 patch
  with tf.variable_scope('conv_layer1'):
    h_conv1 = layers.convolution(
        feature, 32, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool1 = max_pool_2x2(h_conv1)

  # Second conv layer will compute 64 features for each 5x5 patch.
  with tf.variable_scope('conv_layer2'):
    h_conv2 = layers.convolution(
        h_pool1, 64, kernel_size=[5, 5], activation_fn=tf.nn.relu)
    h_pool2 = max_pool_2x2(h_conv2)
    # reshape tensor into a batch of vectors
    h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])

  # Densely connected layer with 1024 neurons.
  h_fc1 = layers.dropout(
      layers.fully_connected(
          h_pool2_flat, 1024, activation_fn=tf.nn.relu),
      keep_prob=0.5,
      is_training=mode == tf.contrib.learn.ModeKeys.TRAIN)

  # Compute logits (1 per class) and compute loss.
  logits = layers.fully_connected(h_fc1, 10, activation_fn=None)
  loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

  # Create a tensor for training op.
  train_op = layers.optimize_loss(
      loss,
      tf.contrib.framework.get_global_step(),
      optimizer='SGD',
      learning_rate=0.001)

  return tf.argmax(logits, 1), loss, train_op


def main(img):

    model_dir_name = 'my_model'

    ### Download and load MNIST dataset.
    print(Fore.GREEN + 'Downloading the MNIST dataset...' + Style.RESET_ALL)
    mnist = learn.datasets.load_dataset('mnist')

    ### Convolutional network
    classifier = learn.Estimator(model_fn=conv_model, model_dir='./' + model_dir_name)

    # If the model is already trained change steps to zero
    if os.path.isdir('./' + model_dir_name):
        steps = 0
        t0 = time()
        classifier.fit(mnist.train.images, mnist.train.labels, batch_size=100, steps=steps)
        print(Fore.YELLOW + "Model already trained\nLoading model" + Style.RESET_ALL)
    else:
        steps = 2000
        print(Fore.GREEN + 'The model is in training mode' + Style.RESET_ALL)
        t0 = time()
        classifier.fit(mnist.train.images, mnist.train.labels, batch_size=100, steps=steps)
        print(Fore.GREEN + "Training time: " + str(round(time()-t0, 3)) +  "s" + Style.RESET_ALL)



    ### First predection to calculate the model accuracy
    pred = classifier.predict(mnist.test.images)
    accuracy_score = metrics.accuracy_score(mnist.test.labels, list(pred)) * 100
    print(Fore.GREEN + 'Accuracy Score: ' + str(accuracy_score) + ' %' + Style.RESET_ALL)

    while True:
        ### Select the image to predict
        img_index = int(raw_input('Index: '))
        label = mnist.test.labels[img_index]
        image = mnist.test.images[img_index]

        t0 = time()
        pred = classifier.predict(image)
        print(Fore.GREEN + "Prediction time: " + str(round(time()-t0, 3)) +  "s" + Style.RESET_ALL)

        ### Select and reshape to 28x28 the image the show
        label_pred = list(pred)[0]
        img = image.reshape((28, 28))

        print(Fore.BLUE + 'Excepted: ', label, Style.RESET_ALL)
        print(Fore.MAGENTA + 'Predicted: ', label_pred, Style.RESET_ALL)

        ### Show the predicted number
        plt.title('Excepted: {label} \n Predicted: {pred}'.format(label=label, pred=label_pred))
        plt.imshow(img, cmap='gray')
        plt.show()


if __name__ == '__main__':
    # Enable tensorflow logging
    tf.logging.set_verbosity(tf.logging.INFO)

    # Init Colorama for the FANCY terminal stuff
    colorama.init()

    # Run tensorflow session
    tf.app.run()
