#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy

from model.complex.architecture import CropFieldHealthModel
from model.light.light_network import light_model, light_model_v2

def load_model(dtype = 'light', weights = None, compile = True):
   """Load the Crop Field Health model w/ or w/o weights."""
   if not isinstance(weights, str): # Validate weights (if provided).
      raise TypeError("The weights argument should be a string containing the "
                      f"model name or filepath, got {type(weights)}.")

   # Determine whether argument weights is a filepath or a file name.
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

   # Initialize model, load weights if required to.
   if dtype == 'crop':
      model = CropFieldHealthModel()
   elif dtype == 'light':
      model = light_model()
   elif dtype == 'lightv2':
      model = light_model_v2()
   else:
      raise ValueError("Invalid model type recieved, should be 'crop' or 'light'.")

   if weights:
      try:
         model.load_weights(weights_file)
      except Exception as e:
         print("Weights file invalid for CropFieldHealthModel, try a different weights file.")
         raise e

   # Compile model if requested to.
   if compile:
      if compile == 'default':
         model.compile(
            optimizer = Adam(),
            loss = CategoricalCrossentropy(),
            metrics = ['accuracy']
         )
      if isinstance(compile, (list, tuple)):
         try:
            model.compile(
               optimizer = compile[0],
               loss = compile[1],
               metrics = ['accuracy']
            )
         except Exception as e:
            raise e

   # Return model.
   return model

