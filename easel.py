"""Deep Learning model for de-novo genome annotation built with Keras.

"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


WEIGHTS_PATH = "" # TBD, for using transfer-learning for cross-species gene prediction
WEIGHTS_PATH_NO_TOP = ""

def conv2d(x, filters, num_row, num_col, padding='same', strides=(1, 1)):
    """Utility function to apply conv
    """
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=False)(x)
    x = layers.Activation('relu')(x)
    return x



def easel(classes=2,
        **kwargs):
    """Initializes the model.

    Can be extended to load pre-trained weights.

    # Returns
        A Keras model instance

    # Raises
        ValueError: for invalid input shape
    """

    input_shape = (128,128)
    img_input = layers.Input(shape=input_shape)

    # Initial Conv + pooling

    x = conv2d(x, 32, 3, 3, strides=(2,2))
    x = conv2d(x, 32, 3, 3)
    x = layers.MaxPooling2D((2,2),strides=(2,2))(x)

    # Inception block 1
    branch1x1 = conv2d(x, 64, 1, 1)

    branch3x3dbl = conv2d(x, 64, 1, 1)
    branch3x3dbl = conv2d(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed1')


    # Inception block 2
    branch1x1 = conv2d(x, 64, 1, 1)

    branch5x5 = conv2d(x, 48, 1, 1)
    branch5x5 = conv2d(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d(x, 64, 1, 1)
    branch3x3dbl = conv2d(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d(branch_pool, 64, 1, 1)
    x = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool],
        axis=channel_axis,
        name='mixed2')

    # Classify block
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(classes,activation='softmax', name='predictions')(x)

    inputs = img_input

    # Create model
    model = keras.Model(inputs, x, name='easel')


    # Loading weights would be done here (TBD)


    return model








