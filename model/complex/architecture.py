#!/usr/bin/env python3

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.layers import Input, Dropout, Add, Concatenate

from model.complex.blocks import make_divisible
from model.complex.blocks import convolution_block, separable_convolution_block, upsample_block
from model.complex.blocks import inverted_resnet_block

__all__ = ['CropFieldHealthModel']

def ModifiedMobileNet(input_shape = (512, 512, 4), classes = 8, final_filters = 64):
   """A modified MobileNetv2 encoder architecture."""
   input = Input(input_shape)

   # Set up input filters.
   input_filters = make_divisible(32, 8)

   # First convolution block.
   x = convolution_block(input, input_filters, kernel_size = (3, 3), strides = (2, 2),
                         padding = 'same', block_name = 'initial')

   # First inverted ResNet block.
   x = inverted_resnet_block(x, expansion = 1, strides = 1, filters = 16, block_id = 0)

   # Second inverted ResNet block.
   x = inverted_resnet_block(x, expansion = 6, strides = 2, filters = 24, block_id = 1)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 24, block_id = 2)

   # Third inverted ResNet block.
   x = inverted_resnet_block(x, expansion = 6, strides = 2, filters = 32, block_id = 3)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 32, block_id = 4)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 32, block_id = 5)

   # Fourth inverted ResNet block.
   x = inverted_resnet_block(x, expansion = 6, strides = 2, filters = 64, block_id = 6)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 64, block_id = 7)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 64, block_id = 8)
   x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 64, block_id = 9)

   # Fifth inverted ResNet block.
   # x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 96, block_id = 10)
   # x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 96, block_id = 11)
   # x = inverted_resnet_block(x, expansion = 6, strides = 1, filters = 96, block_id = 12)

   # Final convolution block.
   x = convolution_block(x, final_filters, kernel_size = (1, 1), block_name = 'final')

   # Create model.
   model = Model(input, x, name = 'Encoder')

   return model

def CropFieldHealthModel(input_shape = (512, 512, 4), classes = 8):
   """Built the complete encoder-decoder model for image segmentation."""
   base = ModifiedMobileNet(input_shape, classes = classes, final_filters = 64)
   unet_layers = ['block_1_expand_relu', 'block_3_expand_relu', 'block_5_expand_relu', 'final_relu']
   layers = [base.get_layer(name).output for name in unet_layers]
   encoder_stack = Model(base.input, layers)
   input = Input(input_shape)
   x = input

   # Downsampling through the encoder stack model.
   encoder = encoder_stack(input)
   x = encoder[-1]
   x = Dropout(0.2)(x)

   # Upsampling through a decoder model.
   upsample_shape = x.shape[1:3]
   upsample_x_1 = GlobalAveragePooling2D()(x)
   upsample_x_1 = tf.expand_dims(tf.expand_dims(upsample_x_1, 1), 1)
   upsample_x_1 = upsample_block(upsample_x_1, 64, kernel_size = (1, 1), upsample = upsample_shape)

   # Split into five branches.
   upsample_conv_1 = convolution_block(x, 64, kernel_size = (1, 1))
   upsample_conv_2 = separable_convolution_block(x, 64, rate = 6)
   upsample_conv_3 = separable_convolution_block(x, 64, rate = 12)
   upsample_conv_4 = separable_convolution_block(x, 64, rate = 18)
   upsample_concat_1 = Concatenate()([upsample_x_1, upsample_conv_1, upsample_conv_2, upsample_conv_3, upsample_conv_4])

   x = upsample_block(upsample_concat_1, 64, kernel_size = (1, 1))
   upsample_temp_branch = convolution_block(encoder[-2], 64, kernel_size = (1, 1))
   x = Concatenate()([x, upsample_temp_branch])
   x = upsample_block(x, 64, kernel_size = (3, 3))
   upsample_temp_branch = convolution_block(encoder[-3], 64, kernel_size = (3, 3))
   x = Concatenate()([x, upsample_temp_branch])

   # Final convolutions and upsampling.
   x_final_conv = convolution_block(x, 8, kernel_size = (3, 3))
   x_final_conv = convolution_block(x_final_conv, 8, kernel_size = (3, 3))
   x_final_conv = upsample_block(x_final_conv, 8, kernel_size = (3, 3), upsample = 4)
   x_final_upsample = upsample_block(x, 8, kernel_size = (3, 3), upsample = 4)
   x = Add()([x_final_conv, x_final_upsample])

   return Model(input, x)

