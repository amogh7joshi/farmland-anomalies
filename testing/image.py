#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np
import tensorflow as tf

from testing.process import preprocess_image
from preprocessing.dataset import AgricultureVisionDataset

def get_testing_image(mode, value = None):
   """Load a testing image, either from a provided image path or from a dataset."""
   if mode and value is None:
      # In this case, an image path has been passed.
      if not os.path.exists(mode):
         raise FileNotFoundError("You have only provided an argument for `mode`, this case expects "
                                 "an image filepath. Try passing a different, valid image path.")
      else:
         # Return the processed image.
         return preprocess_image(mode)
   else:
      # Choose the dataset.
      if mode not in ["train", "val", "test", "eval"]:
         raise ValueError(f"Received invalid dataset mode '{mode}', expecting train, val, test, or eval.")
      # Construct the dataset.
      dataset = AgricultureVisionDataset()
      dataset.construct()

      # Get the corresponding dataset.
      if mode == "eval":
         iterator = dataset.evaluation_dataset()
      else:
         iterator = getattr(dataset, f"{mode}_data")

      # Iterate over the data.
      for indx, item in enumerate(iterator):
         # If the index is equal to the value.
         if indx == value:
            if hasattr(item, "numpy"):
               # Item is just a pure piece of image data (from a test set).
               return tf.expand_dims(item.numpy()[0], axis = 0)
            elif len(item) == 2:
               # Item is a train/label data (from train/val).
               return tf.expand_dims(item[0].numpy()[0], axis = 0)
            else:
               # Any other case which has not already been covered.
               return tf.expand_dims(item[0], axis = 0)
         else:
            continue

   # For some reason, if nothing has been returned, then throw an error.
   raise Exception("Nothing was returned, something must be broken.")

def create_displayable_test_output(test_image):
   """Processes an (already post-processed) test image to a displayable format."""
   return cv2.cvtColor(np.squeeze(test_image)[:, :, 3], cv2.COLOR_BGR2RGB)

