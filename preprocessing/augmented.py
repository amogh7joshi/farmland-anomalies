#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
import argparse

import numpy as np
from matplotlib import style
style.use('seaborn-dark')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec

import tensorflow as tf

from preprocessing.dataset import AgricultureVisionDataset

# Plots augmented images (after images have been processed in the dataset).
dataset = AgricultureVisionDataset(augmentation = True)
dataset.construct()

def get_classes_dict(dataset_dir):
   """Get a dictionary of classes for the Agriculture-Vision dataset."""
   class_dict = {0: 'background'}; count = 1
   label_path = os.path.join(dataset_dir, 'train', 'labels')
   for item in os.listdir(label_path):
      if item == '.DS_Store' and sys.platform == 'darwin':
         continue # Skip .DS_Store on MacOS.
      class_dict[count] = str(item)
      count += 1
   return class_dict

def plot_augmented_single_example_images(num = 0, background = "light"):
   """Plots a single row of augmented images from the dataset."""
   # Get the first item of the dataset.
   for indx, item in enumerate(iter(dataset.evaluation_dataset(1))):
      # We only need a single item (but have to
      # use a for loop because of __iter__ issues).
      if indx != num:
         if indx > num:
            break
         continue

      # Get the array representation of an EagerTensor for ease in plotting.
      # image, label = item[0].numpy(), item[1].numpy()
      image, label = item[0], item[1]

      # Remove the now-irrelevant batch axis.
      image, label = np.squeeze(image), np.squeeze(label)

      # Unpack the `image` into the RGB and NIR images.
      rgb_image = image[:, :, :3]
      nir_image = image[:, :, 3]

      # Unpack the `label` into the background, labels, and invalid pixels.
      background_image = label[:, :, 1]
      label_images = label[:, :, 1: 7]
      label_images = np.moveaxis(label_images, -1, 0)
      invalid_pixels = label[:, :, 7]

      # Create the figure.
      fig, axes = plt.subplots(1, 9, figsize = (20, 5))
      if background == "dark":
         fig.patch.set_facecolor('#2e3037ff')
      elif background == 'light':
         fig.patch.set_facecolor('#efefefff')
      for i, ax in enumerate(axes.flat):
         # Turn off axis.
         ax.axis('off')

         # Plot the RGB image.
         if i == 0:
            ax.imshow(rgb_image, vmin = 0, vmax = 255, cmap = 'magma')
            image_type_title = 'RGB'
         # Plot the NIR image.
         elif i == 1:
            ax.imshow(background_image, cmap = 'gray')
            image_type_title = 'Background'
         # Plot the label images.
         elif i in list(range(2, 8)):
            ax.imshow(label_images[i - 2], cmap = 'gray')
            image_type_title = get_classes_dict('../data/Agriculture-Vision')[i - 1].title()
         # Plot the invalid pixels.
         elif i == 8:
            ax.imshow(invalid_pixels, cmap = 'gray')
            image_type_title = 'Invalid Pixels'

         # Set title.
         if background == "dark":
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 18, color = 'w')
         else:
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 18, color = 'k')

      # Display plot.
      plt.show()

if __name__ == '__main__':
   # Parse command line arguments.
   ap = argparse.ArgumentParser()
   ap.add_argument('-v', '--val', default = 9, help = "The value of the image that you want to display.")
   args = ap.parse_args()

   # Run the main method.
   plot_augmented_single_example_images(args.val)

