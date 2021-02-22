#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, DepthwiseConv2D, AveragePooling2D
from tensorflow.keras.layers import BatchNormalization, MaxPooling2D, UpSampling2D
from tensorflow.keras.layers import InputLayer, Input, Add, SpatialDropout2D, Concatenate, Dropout
from tensorflow.keras.regularizers import l2

__all__ = ['light_model', 'light_model_v2']

def light_model(input_shape = (512, 512, 4), classes = 8):
   """Construct the encoder-decoder model architecture."""
   input = Input(input_shape)

   # First Encoder stage 1.
   enc = Conv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(input)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # First Encoder stage 2.
   enc = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # First Encoder stage 3.
   enc = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # First Encoder stage 4.
   enc = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', kernel_regularizer = l2(0.01))(enc)
   enc = BatchNormalization()(enc)
   enc = MaxPooling2D(pool_size = (2, 2), padding = 'same')(enc)
   enc = Dropout(0.1)(enc)

   # Second Encoder stage 1.
   enc2 = SeparableConv2D(256, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal')(input)
   enc2 = BatchNormalization()(enc2)
   enc2 = AveragePooling2D(pool_size = (2, 2), padding = 'same')(enc2)
   enc2 = Dropout(0.1)(enc2)

   # Second Encoder stage 2.
   enc2 = SeparableConv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal')(enc2)
   enc2 = BatchNormalization()(enc2)
   enc2 = AveragePooling2D(pool_size = (2, 2), padding = 'same')(enc2)
   enc2 = Dropout(0.1)(enc2)

   # Second Encoder stage 3.
   enc2 = SeparableConv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal')(enc2)
   enc2 = BatchNormalization()(enc2)
   enc2 = AveragePooling2D(pool_size = (2, 2), padding = 'same')(enc2)
   enc2 = Dropout(0.1)(enc2)

   # Second Encoder stage 4.
   enc2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal')(enc2)
   enc2 = BatchNormalization()(enc2)
   enc2 = AveragePooling2D(pool_size = (2, 2), padding = 'same')(enc2)
   enc2 = Dropout(0.1)(enc2)

   # Concatenate encoders into a single output.
   encoder_output = Concatenate()([enc, enc2])

   # Decoder stage 1.
   dec = Conv2D(32, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(encoder_output)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder stage 2.
   dec = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder Stage 3.
   dec = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   dec = Dropout(0.1)(dec)

   # Decoder Stage 4.
   dec = Conv2D(32, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal')(dec)
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)
   decoder_output = Dropout(0.1)(dec)

   # Model output stage.
   output = Conv2D(classes, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same',
                   kernel_initializer = 'he_normal')(decoder_output)

   return Model(input, output)

def light_model_v2(input_shape = (512, 512, 4), classes = 8):
   """Construct the encoder-decoder model architecture."""
   input = Input(input_shape)

   # First Encoder stage 1.
   enc = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', strides = (2, 2))(input)
   enc = BatchNormalization()(enc)

   # First Encoder stage 2.
   enc = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', strides = (2, 2))(enc)
   enc = BatchNormalization()(enc)

   # First Encoder stage 3.
   enc = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', strides = (2, 2))(enc)
   enc = BatchNormalization()(enc)

   # First Encoder stage 4.
   enc = Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                kernel_initializer = 'he_normal', strides = (2, 2))(enc)
   enc = BatchNormalization()(enc)

   # Second Encoder stage 1.
   enc2 = SeparableConv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal', strides = (2, 2))(input)
   enc2 = BatchNormalization()(enc2)

   # Second Encoder stage 2.
   enc2 = SeparableConv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal', strides = (2, 2))(enc2)
   enc2 = BatchNormalization()(enc2)

   # Second Encoder stage 3.
   enc2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal', strides = (2, 2))(enc2)
   enc2 = BatchNormalization()(enc2)

   # Second Encoder stage 4.
   enc2 = SeparableConv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                          kernel_initializer = 'he_normal', strides = (2, 2))(enc2)
   enc2 = BatchNormalization()(enc2)

   # Concatenate encoders into a single output.
   encoder_output = Concatenate()([enc, enc2])

   # Decoder stage 1.
   dec_branch_1 = Conv2D(32, kernel_size = (5, 5), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal')(encoder_output)
   dec_branch_2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                  kernel_initializer = 'he_normal')(encoder_output)
   dec = Add()([dec_branch_1, dec_branch_2])
   dec = UpSampling2D(size = (2, 2))(dec)

   # Decoder stage 2.
   dec_branch_1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal')(dec)
   dec_branch_2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                  kernel_initializer = 'he_normal')(dec)
   dec = Add()([dec_branch_1, dec_branch_2])
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)

   # Decoder Stage 3.
   dec_branch_1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal')(dec)
   dec_branch_2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                  kernel_initializer = 'he_normal')(dec)
   dec = Add()([dec_branch_1, dec_branch_2])
   dec = BatchNormalization()(dec)
   dec = UpSampling2D(size = (2, 2))(dec)

   # Decoder Stage 4.
   dec_branch_1 = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                         kernel_initializer = 'he_normal')(dec)
   dec_branch_2 = SeparableConv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                  kernel_initializer = 'he_normal')(dec)
   dec = Add()([dec_branch_1, dec_branch_2])
   dec = BatchNormalization()(dec)
   decoder_output = UpSampling2D(size = (2, 2))(dec)

   # Mini-encoder-decoder to learn higher-level features.
   # MIni-encoder-decoder encoding branch 1.
   mini_model_branch1 = DepthwiseConv2D(kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                        kernel_initializer = 'he_normal')(input)
   mini_model_branch1 = Conv2D(128, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(mini_model_branch1)
   mini_model_branch1 = BatchNormalization()(mini_model_branch1)
   mini_model_branch1 = MaxPooling2D(pool_size = (2, 2))(mini_model_branch1)

   mini_model_branch1 = DepthwiseConv2D(kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                        kernel_initializer = 'he_normal')(mini_model_branch1)
   mini_model_branch1 = Conv2D(64, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(mini_model_branch1)
   mini_model_branch1 = BatchNormalization()(mini_model_branch1)
   mini_model_branch1 = MaxPooling2D(pool_size = (2, 2))(mini_model_branch1)

   mini_model_branch1 = DepthwiseConv2D(kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                        kernel_initializer = 'he_normal')(mini_model_branch1)
   mini_model_branch1 = Conv2D(16, kernel_size = (1, 1), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(mini_model_branch1)
   mini_model_branch1 = BatchNormalization()(mini_model_branch1)
   mini_model_branch1 = MaxPooling2D(pool_size = (2, 2))(mini_model_branch1)

   # Mini encoder-decoder encoding branch 2.
   mini_model_branch2 = Conv2D(128, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(input)
   mini_model_branch2 = BatchNormalization()(mini_model_branch2)
   mini_model_branch2 = AveragePooling2D(pool_size = (2, 2))(mini_model_branch2)

   mini_model_branch2 = Conv2D(64, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(mini_model_branch2)
   mini_model_branch2 = BatchNormalization()(mini_model_branch2)
   mini_model_branch2 = AveragePooling2D(pool_size = (2, 2))(mini_model_branch2)

   mini_model_branch2 = Conv2D(16, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                               kernel_initializer = 'he_normal')(mini_model_branch2)
   mini_model_branch2 = BatchNormalization()(mini_model_branch2)
   mini_model_branch2 = AveragePooling2D(pool_size = (2, 2))(mini_model_branch2)

   # Concatenate encoding branches.
   mini_model_encoding = Concatenate()([mini_model_branch1, mini_model_branch2])

   # Mini encoder-decoder decoder segment.
   mini_model_decoding = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                kernel_initializer = 'he_normal')(mini_model_encoding)
   mini_model_decoding = BatchNormalization()(mini_model_decoding)
   mini_model_decoding = UpSampling2D(size = (2, 2))(mini_model_decoding)

   mini_model_decoding = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                kernel_initializer = 'he_normal')(mini_model_decoding)
   mini_model_decoding = BatchNormalization()(mini_model_decoding)
   mini_model_decoding = UpSampling2D(size = (2, 2))(mini_model_decoding)

   mini_model_decoding = Conv2D(32, kernel_size = (3, 3), activation = 'relu', padding = 'same',
                                kernel_initializer = 'he_normal')(mini_model_decoding)
   mini_model_decoding = BatchNormalization()(mini_model_decoding)
   mini_model_decoding = UpSampling2D(size = (2, 2))(mini_model_decoding)

   # Concatenate primary model and mini-model.
   decoder_output = Add()([mini_model_decoding, decoder_output])

   # Model output stage.
   output = Conv2D(classes, kernel_size = (3, 3), activation = 'sigmoid', padding = 'same',
                   kernel_initializer = 'he_normal')(decoder_output)

   return Model(input, output)


