#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import itertools
import argparse

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf

from preprocessing.dataset import AgricultureVisionDataset
from model.model_factory import load_model
from tensorflow.keras.models import load_model
# Construct Dataset.
dataset = AgricultureVisionDataset()
dataset.construct()
evaluation_data = dataset.evaluation_dataset(batch = 1)

# Load model.
model = load_model('../data/models/Model-08-0.8824.hdf5')

# Construct list of evaluation classes.
evaluation_classes = ['Background', 'Waterway', 'Standing Water', 'Weed Cluster',
                      'Double Plant', 'Planter Skip', 'Cloud Shadow', 'Invalid Pixels']

# Iterate over validation data.
def evaluate_validation_image(num = 10, mode = 'save'):
   """Show or save to image an evaluation of the model (on an evaluation image)."""
   if mode not in ['save', 'show']:
      raise ValueError("Invalid model provided, should be either 'save' or 'show'.")
   if not isinstance(num, int):
      raise TypeError(f"Number of images to evaluate an integer, got {type(num)}.")

   # Construct figure.
   for i, (x, y) in enumerate(itertools.islice(evaluation_data, num)):
      # Predict output from model.
      y_predicted = model.predict(x)
      print(f"Creating figure {i + 1} of {num}.")

      # Create figure.
      fig, axes = plt.subplots(2, 9, figsize = (20, 6), constrained_layout = True)

      # Display images in figure.
      for indx, ax in enumerate(axes.flat):
         if indx == 0: # Show original image (rgb).
            ax.set_title('Original Image', fontdict = {'fontsize': 14})
            ax.imshow(x[0, :, :, 1:])
         elif indx < 9: # Show labels and masks.
            ax.set_title(dataset.class_list[indx - 1], fontdict = {'fontsize': 14})
            ax.imshow(y[0, :, :, indx - 1], cmap = 'gray')
         elif indx == 9: # Show original image (nir).
            ax.imshow(x[0, :, :, 0])
         else: # Show predicted images.
            ax.imshow(y_predicted[0, :, :, (indx - 2) % 8], vmin = 0, vmax = 1, cmap = 'gray')

         # Remove tick labels.
         ax.set_xticklabels([])
         ax.set_yticklabels([])

         # Format ax.
         ax.set_aspect('equal')

      # Set y-axis to differentiate between real and predicted.
      for ax, row in zip(axes[:, 0], ['Truth', 'Prediction']):
         ax.set_ylabel(row, rotation = 90, size = 'xx-large')

      # Adjust figure.
      # fig.subplots_adjust(hspace = 0, wspace = 0.15)

      # Save figure.
      if mode == 'save':
         savefig = plt.gcf()
         save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', f'evaluated-{i + 1}.png')
         if os.path.exists(save_path):
            os.remove(save_path)
         savefig.savefig(save_path)

      # Show figure.
      if mode == 'show':
         plt.show()

if __name__ == '__main__':
   # Construct parser and parse command line arguments.
   ap = argparse.ArgumentParser()
   ap.add_argument('--num-images', default = 10, type = int,
                   help = 'The number of images to evaluate.')
   ap.add_argument('--mode', default = 'save', type = str,
                   help = 'The evaluation mode: either show figures or save to image files.')
   args = ap.parse_args()

   # Execute evaluation.
   evaluate_validation_image(args.num_images, args.mode)




