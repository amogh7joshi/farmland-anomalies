#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import warnings

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, DepthwiseConv2D, MaxPooling2D, UpSampling2D, ZeroPadding2D
from tensorflow.keras.layers import InputLayer, ReLU, Dropout, Add
from tensorflow.keras.regularizers import l2
from keras_applications import correct_pad

# Helper functions.

def validate_image_format(func):
   """Decorator to validate image_data_format."""
   def inner(*args, **kwargs):
      # Confirm image_data_format.
      if K.image_data_format() != 'channels_last':
         warnings.warn('You should be using a TensorFlow backend, which uses channel_last.', SyntaxWarning)
      return func(*args, **kwargs)
   return inner

def make_divisible(value, divisor, min_value = None):
   """Ensures that each layer has a channel number divisible by 8.
   Taken from the original tensorflow repository, and can be seen at
   https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py"""
   if min_value is None:
      min_value = divisor
   new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
   if new_value < 0.9 * value:
      new_value += divisor
   return new_value

# Primary architecture functions.

@validate_image_format
def convolution_block(input, filters, kernel_size, strides = (1, 1), padding = 'same',
                      activation = 'relu', use_bias = False, block_name = None):
   """A convolution block: Convolution, BatchNormalization, and ReLU activation."""
   if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)

   with tf.name_scope('convolution_block'):
      if block_name:
         x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_regularizer = l2(0.01),
                    kernel_initializer = 'he_normal', use_bias = use_bias, name = f'{block_name}_conv')(input)
         x = BatchNormalization(momentum = 0.999, name = f'{block_name}_bn')(x)
         x = ReLU(name = f'{block_name}_relu')(x)
      else:
         x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, kernel_regularizer = l2(0.01),
                    kernel_initializer = 'he_normal', use_bias = use_bias)(input)
         x = BatchNormalization(momentum = 0.999)(x)
         x = ReLU()(x)

   return x

@validate_image_format
def separable_convolution_block(input, filters, rate = 1, kernel_size = (3, 3), strides = (1, 1)):
   """Depthwise separable Convolution Block: DepthwiseConvolution & Convolution."""
   if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
   depth_padding = 'same'

   with tf.name_scope('separable_convolution_block'):
      x = DepthwiseConv2D(kernel_size = kernel_size, strides = strides, dilation_rate = (rate, rate),
                          padding = depth_padding, use_bias = False)(input)
      x = BatchNormalization(momentum = 0.999)(x)
      x = ReLU()(x)
      x = convolution_block(x, filters, kernel_size = (1, 1), padding = 'same')

   return x

@validate_image_format
def upsample_block(input, filters, kernel_size, strides = (1, 1), padding = 'same', upsample = 2):
   """An upsampling block: Convolution and then Upsampling."""
   if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
   if isinstance(upsample, int):
      upsample = (upsample, upsample)

   with tf.name_scope('upsampling_block'):
      x = Conv2D(filters, kernel_size = kernel_size, strides = strides, padding = padding, activation = 'relu',
                 kernel_initializer = 'he_normal')(input)
      x = BatchNormalization()(x)
      x = UpSampling2D(size = upsample, interpolation = 'bilinear')(x)

   return x

@validate_image_format
def inverted_resnet_block(input, expansion, strides, filters, block_id):
   """Inverted ResNet block. Modified from tf.keras.applications.mobilenet_v2"""
   channels = input.shape[-1]
   pointwise_filters = make_divisible(int(filters), 8)

   with tf.name_scope('inverted_resnet_block'):
      # Expansion.
      x = convolution_block(input, expansion * channels, kernel_size = (1, 1), padding = 'same',
                            activation = 'relu', use_bias = False, block_name = f'block_{block_id}_expand')
      # Depthwise.
      x = DepthwiseConv2D(kernel_size = (3, 3), strides = strides, padding = 'same',
                          use_bias = False, name = f'block_{block_id}_depthwise_conv')(x)
      x = BatchNormalization(name = f'block_{block_id}_depthwise_bn')(x)
      x = ReLU(name = f'block_{block_id}_depthwise_relu')(x)
      # Projection.
      x = convolution_block(x, pointwise_filters, kernel_size = (1, 1), padding = 'same',
                            activation = 'relu', use_bias = False, block_name = f'block_{block_id}_project')
      if channels == pointwise_filters and strides == 1: # Skip connection.
         x = Add()([input, x])

   return x




