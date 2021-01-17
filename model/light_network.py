#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import InputLayer, Input, Dense, Dropout
from tensorflow.keras.regularizers import l2

__all__ = ['light_model']

def light_model(input_shape = (512, 512, 4), classes = 8):
   """Construct the encoder-decoder model architecture."""
   input = Input(input_shape)

   # Encoder stage 1.
   enc = Conv2D(128, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = Conv2D(128, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # Encoder stage 2.
   enc = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # Encoder stage 3.
   enc = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # Encoder stage 4.
   enc = Conv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = Conv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # Decoder stage 1.
   dec = SeparableConv2D(32, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   dec = BatchNormalization()(dec)
   dec = Conv2D(32, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder stage 2.
   dec = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(dec)
   dec = BatchNormalization()(dec)
   dec = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder Stage 3.
   dec = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(dec)
   dec = BatchNormalization()(dec)
   dec = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder Stage 4.
   dec = SeparableConv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(dec)
   dec = BatchNormalization()(dec)
   dec = Conv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Model output stage.
   output = Conv2D(classes, kernel_size = (3, 3), activation = 'softmax', padding = 'same',
                   kernel_initializer = 'he_normal')(dec)

   return Model(input, output)



