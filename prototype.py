#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import json

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from model.loss import dice_loss_2d, surface_channel_loss_2d
from preprocessing.dataset import AgricultureVisionDataset
from testing.image import get_testing_image
from testing.process import preprocess_image, postprocess_output
from testing.segment import draw_segmentation_map, display_segmented_pair

class FarmlandAnomalyModel(object):
   """The complete container class for the farmland anomaly segmentation model."""
   def __init__(self, model = 'final', **kwargs):
      # Set up the actual model which is being stored in the container.
      self.model = None
      self._initialize_model(model, **kwargs)

   def __str__(self):
      # Print out the summary of the model architecture.
      try:
         print(self.model.summary())
      except TypeError:
         # Technically, this method doesn't actually return anything, it just prints
         # out a summary. However, it's nice to have an easily displayable way to view
         # the model architecture, so we just capture the error and move on here.
         pass
      finally:
         # To subjugate any TypeErrors which may arise in the future.
         return ''

   def _initialize_model(self, model, **kwargs):
      """Initializes the model from a provided argument or weights path."""
      # First, determine if the argument is actually a path.
      if model.endswith('.hdf5') or model.endswith('.h5'):
         # Validate the path and set the class argument.
         if os.path.exists(model):
            # Determine if custom objects are necessary.
            if 'custom_objects' in kwargs.keys():
               # Validate the custom objects first.
               custom_objects = self._validate_custom_objects(kwargs['custom_objects'])
               self.model = load_model(model, custom_objects = custom_objects)
            else:
               # Otherwise, just load the model.
               self.model = load_model(model)
      else:
         # Otherwise, we've gotten a shortcut model name in which case load that specific one.
         if model.lower() in ['dice20', '20', 'first', 'stage1']:
            self.model = load_model(os.path.join(os.path.dirname(__file__), 'logs/save/Model-Dice2D-20.hdf5'),
                                    custom_objects = self._validate_custom_objects('dice'))
         elif model.lower() in ['scl40', '40', 'middle', 'stage2', 'intermediate']:
            self.model = load_model(os.path.join(os.path.dirname(__file__), 'logs/save/Model-Dice-SCL-40.hdf5'),
                                    custom_objects = self._validate_custom_objects('scl'))
         elif model.lower() in ['dice60', '60', 'last', 'final', 'stage3']:
            self.model = load_model(os.path.join(os.path.dirname(__file__), 'logs/save/Model-Dice-SCL-Dice-60.hdf5'),
                                    custom_objects = self._validate_custom_objects('dice'))

   @staticmethod
   def _validate_custom_objects(custom_objects):
      """Determine the custom objects of a model, then initialize and set them to the class."""
      if isinstance(custom_objects, dict):
         # If the provided item is a dictionary of objects, then simply return the
         # dictionary of custom objects; there is no processing to do here.
         return custom_objects
      else:
         # Otherwise, a shortcut name has been provided for a loss function, in which
         # case we need to determine and validate the actual provided function.
         if custom_objects.lower() == 'dice':
            return {'dice_loss_2d': dice_loss_2d}
         elif custom_objects.lower() == 'scl' or custom_objects.lower() == 'surface':
            return {'surface_loss_2d': surface_channel_loss_2d}
         else:
            raise ValueError(f"Received invalid custom object shortcut keyword {custom_objects}.")

   def predict(self, test_image):
      """Predicts the output of a provided input image.

      The image can be either directly from the dataset, or be a provided image path/read image.

      Returns the segmented 8-channel image.

      The method `show_segmented_predictions` displays the actual segmented prediction maps,
      and the method `show_channel_predictions` shows the channel-by-channel predictions.

      Parameters:
         - test_image: The inputted test image (see above for information about the image format).
      Returns:
         - The 8-channel prediction image.
      """
      # Get a valid testing image.
      if isinstance(test_image, (list, tuple, set)):
         # If we're using an item from the dataset, then get the specific dataset item.
         testing_image = get_testing_image(mode = test_image[0], value = test_image[1], with_truth = False)
      else:
         # Otherwise, preprocess the input image.
         testing_image = preprocess_image(test_image)

      # Now, predict the output from the model and return postprocessed predictions.
      return postprocess_output(self.model.predict(testing_image))




