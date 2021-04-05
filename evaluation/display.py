#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import itertools
import argparse

import cv2
import numpy as np
from matplotlib import style
style.use('seaborn-dark')
import matplotlib.pyplot as plt

import tensorflow as tf

from preprocessing.dataset import AgricultureVisionDataset
from model.model_factory import load_model
from model.loss import surface_channel_loss_2d, dice_loss_2d
from tensorflow.keras.models import load_model

# Construct Dataset.
dataset = AgricultureVisionDataset(augmentation = True)
dataset.construct()
# evaluation_data = dataset.evaluation_dataset(batch = 1)

# Load model.
model = load_model('../logs/save/Model-Dice-SCL-Dice-60.hdf5',
                   custom_objects = {'dice_loss_2d': dice_loss_2d})

# Construct list of evaluation classes.
evaluation_classes = ['Background', 'Waterway', 'Standing Water', 'Weed Cluster',
                      'Double Plant', 'Planter Skip', 'Cloud Shadow', 'Invalid Pixels']

# Iterate over validation data.
def display_validation_image(num = 10, mode = 'save', background = 'light'):
   """Show or save to image an evaluation of the model (on an evaluation image)."""
   if mode not in ['save', 'show']:
      raise ValueError("Invalid model provided, should be either 'save' or 'show'.")
   if not isinstance(num, (int, list, tuple)):
      raise TypeError(f"Expected the number of images to be an integer or a tuple"
                      f"representing the range, got {type(num)}.")

   # If the provided argument is a list/tuple, then only use that subset of the dataset.
   if isinstance(num, (list, tuple)):
      # Create the range.
      valid_indices = list(range(num[0], num[1]))
      num = num[1]
   else:
      valid_indices = False

   # Construct figure.
   for i, (x, y) in enumerate(itertools.islice(dataset.evaluation_dataset(1), num)):
      # Skip if expected to.
      if valid_indices:
         if i not in valid_indices:
            print(f'Skipping Dataset Item {i}.')
            continue

      # Predict output from model.
      y_predicted = model.predict(x)
      print(f"Creating figure {i + 1} of {num}.")

      # Create figure.
      fig, axes = plt.subplots(2, 9, figsize = (20, 6), constrained_layout = True)
      if background == "dark":
         fig.patch.set_facecolor('#2e3037ff')
      elif background == "light":
         fig.patch.set_facecolor('#efefefff')
      elif background == "white":
         fig.patch.set_facecolor('#ffffff')

      # Display images in figure.
      for indx, ax in enumerate(axes.flat):
         if indx == 0: # Show original image (rgb).
            if background == 'dark':
               ax.set_title('Original Image', fontsize = 14, color = 'w')
            else:
               ax.set_title('Original Image', fontsize = 14, color = 'k')
            ax.imshow(x[0, :, :, 1:])
         elif indx < 9: # Show labels and masks.
            if background == 'dark':
               ax.set_title(dataset.class_list[indx - 1], fontsize = 16, color = 'w')
            else:
               ax.set_title(dataset.class_list[indx - 1], fontsize = 16, color = 'k')
            ax.imshow(y[0, :, :, indx - 1], cmap = 'gray')
         elif indx == 9: # Show original image (nir).
            ax.imshow(x[0, :, :, 0])
         else: # Show predicted images.
            ax.imshow(y_predicted[0, :, :, (indx - 2) % 8], vmin = 0, vmax = 1, cmap = 'gray')

         # Turn off the axis.
         ax.axis('off')

         # Remove tick labels.
         ax.set_xticklabels([])
         ax.set_yticklabels([])

         # Format ax.
         ax.set_aspect('equal')

      # Set y-axis to differentiate between real and predicted.
      for ax, row in zip(axes[:, 0], ['Truth', 'Prediction']):
         ax.set_ylabel(row, rotation = 90, size = 'xx-large', color = 'k')

      # Save figure.
      if mode == 'save':
         savefig = plt.gcf()
         save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                                  'images', f'evaluated-cheese-{i + 1}.png')
         if os.path.exists(save_path):
            os.remove(save_path)
         savefig.savefig(save_path)

      # Show figure.
      if mode == 'show':
         plt.show()

if __name__ == '__main__':
   # Construct parser and parse command line arguments.
   ap = argparse.ArgumentParser()
   ap.add_argument('--num-images', default = 10, type = int or tuple or list,
                   help = 'The number of images to evaluate.')
   ap.add_argument('--mode', default = 'save', type = str,
                   help = 'The evaluation mode: either show figures or save to image files.')
   args = ap.parse_args()

   # Execute evaluation.
   display_validation_image(args.num_images, args.mode, background = "white")




