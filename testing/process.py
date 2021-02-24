#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np
import tensorflow as tf

def preprocess_image(image_path):
   """Preprocesses an image to the Agriculture-Vision dataset format,
   so that it can be used in the developed model."""
   if not os.path.exists(image_path):
      raise FileNotFoundError(f"The provided image file {image_path} does not exist.")

   # Read and resize the image to 512-by-512 pixels.
   image = cv2.imread(image_path)
   image = cv2.resize(image, (512, 512))

   # Create a black-and-white representation of the image, which
   # will be a stand-in for the NIR images of the dataset.
   bw_representation = image.copy()
   bw_representation = cv2.cvtColor(bw_representation, cv2.COLOR_BGR2GRAY)
   bw_representation = np.expand_dims(bw_representation, axis = -1)

   # Concatenate the channels together.
   concat_image = np.concatenate([image, bw_representation], axis = -1)

   # Add the batch channel.
   concat_image = np.expand_dims(concat_image, axis = 0)

   # Reduce pixel regions of image.
   concat_image = concat_image.astype(np.float32)
   concat_image /= 255

   # Return the final image.
   return concat_image

def postprocess_output(prediction):
   """Post-processes a model prediction to a displayable format."""
   prediction = tf.squeeze(prediction)
   prediction = tf.transpose(prediction, [2, 0, 1])
   return prediction.numpy()
