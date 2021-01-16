#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import InputLayer, ReLU, Dropout
from tensorflow.keras.regularizers import l2

# Helper functions.

def validate_image_format(func):
   """Decorator to validate image_data_format."""
   def inner(*args, **kwargs):
      # Confirm image_data_format.
      if K.image_data_format() != 'channels_last':
         warnings.warn('You should be using a TensorFlow backend, which uses channel_last.', SyntaxWarning)
      return func(*args, **kwargs)
   return inner

@validate_image_format
def convolution_block(input, filters, kernel_size, strides = (1, 1), padding = 'same'):
   """A convolution block: Convolution, BatchNormalization, and ReLU activation."""
   if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)

   with tf.name_scope('convolution_block'):
      x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, activation = 'relu',
                 kernel_regularizer = l2(0.01), kernel_initializer = 'he_normal', use_bias = False)(input)
      x = BatchNormalization()(x)

   return x

@validate_image_format
def upsample_block(input, filters, kernel_size, strides = (1, 1), padding = 'same'):
   """An upsampling block: Convolution and then Upsampling."""
   if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)

   with tf.name_scope('upsampling_block'):
      x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, activation = 'relu',
                 kernel_initializer = 'he_normal')(input)
      x = UpSampling2D(size = (2, 2))(x)

   return x


