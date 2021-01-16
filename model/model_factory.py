#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import numpy as np

from tensorflow.keras.models import load_model

from model.architecture import CropFieldHealthModel

def load_crop_model(weights = None):
   """Load the Crop Field Health model w/ or w/o weights."""
   if not isinstance(weights, str):
      raise TypeError("The weights argument should be a string containing the "
                      f"model name or filepath, got {type(weights)}.")
   if weights:
      if weights.endswith('.hdf5') or weights.endswith('.h5'):
         if not os.path.exists(weights):
            raise FileNotFoundError(f"Provided path {weights} does not exist.")
         weights_file = weights
      else:
         weights_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
         weights_file = os.path.join(weights_path, f'{weights}.hdf5')
         if not os.path.exists(weights_file):
            weights_file = os.path.join(weights_path, f'{weights}.h5')
         if not os.path.exists(weights_file):
            raise FileNotFoundError(f"No weights file found for model {weights}.")
   model = CropFieldHealthModel()
   if weights:
      try:
         model.load_weights(weights_file)
      except Exception as e:
         print("Weights file invalid for CropFieldHealthModel, try a different weights file.")
   return model
