#!/usr/bin/env python3
# -*- coding = utf-*-
from __future__ import absolute_import, division, unicode_literals

import os
import sys
import json
import time
import argparse

import cv2
import numpy as np
from matplotlib import style
style.use('seaborn-dark')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gspec

from PIL import Image

from preprocessing.dataset import AgricultureVisionDataset

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

def get_image_names(mode, dataset_dir):
   """Get list of image and directory paths for usage in main generation method."""
   image_ids = []
   if mode not in ['train', 'val', 'test', 'full']:
      raise ValueError("Invalid mode for file and path acquisition: should be train, val, or test.")
   if mode == 'full':
      modes = ['train', 'val', 'test']
   else:
      modes = [mode]

   # Create and validate paths to each individual directory.
   for mode in modes:
      rgb_image_dir = os.path.join(dataset_dir, mode, 'images', 'rgb')
      # Create list of filenames for all images.
      assert os.path.exists(rgb_image_dir), "Path to dataset image feature 'rgb' missing."
      image_files = [name[:-4] for name in os.listdir(rgb_image_dir)]
      image_ids.extend(image_files)

   # Create list of filenames for all images.
   return image_ids

def load_json_files(processed_paths):
   """Load json files with dictionaries of processed paths."""
   global train_data_paths, val_data_paths, test_data_paths, complete_paths
   assert os.path.exists(os.path.join(processed_paths, 'train.json'))
   with open(os.path.join(processed_paths, 'train.json'), 'r') as train_json_file:
      train_data_paths = json.load(train_json_file)
   assert os.path.exists(os.path.join(processed_paths, 'val.json'))
   with open(os.path.join(processed_paths, 'val.json'), 'r') as val_json_file:
      val_data_paths = json.load(val_json_file)
   assert os.path.exists(os.path.join(processed_paths, 'test.json'))
   with open(os.path.join(processed_paths, 'test.json'), 'r') as test_json_file:
      test_data_paths = json.load(test_json_file)

   # Create complete list of paths.
   complete_paths = []
   for item in train_data_paths:
      complete_paths.append(item)
   for item in val_data_paths:
      complete_paths.append(item)
   for item in test_data_paths:
      complete_paths.append(item)

def get_processed_paths(image_id, paths_list):
   """Get processed paths from dictionary."""
   dataset_type_list = paths_list # Set default dict to iterate through.
   for indx, item in enumerate(dataset_type_list):
      if item['id'] == image_id:
         path_subdict = dataset_type_list[indx]
         break
   else: # If the value was not found.
      raise KeyError(f"The item {image_id} was not found in the dataset.")
   return path_subdict

def single_image_info(image_id, category, paths_list):
   """Return image information from a specific category, for show_random_specific_images()."""
   file_paths = get_processed_paths(image_id, paths_list)
   return np.array(Image.open(file_paths[category]))

def all_image_info(image_id, paths_list):
   """Return image information for all categories, for show_random_general_images()."""
   file_paths = get_processed_paths(image_id, paths_list)
   image_info = {}
   for key, value in file_paths.items():
      if key == 'id': continue # Skip key == 'id'.
      if key == 'classes': continue # Skip key == 'classes'.
      image_info[key] = np.array(Image.open(value))
   return image_info

def show_random_specific_images(size, image_ids, image_types, paths_list, save = False):
   """Choose and display a random selection of images from a specific category."""
   assert isinstance(size, (list, tuple)), f"The size parameter should be a list or tuple, got {type(size)}."
   assert len(size) == 2, f"The image size parameter should be 2, got {len(size)}."

   # Choose a random selection of images.
   num_images = size[0] * size[1]
   random_images = np.random.choice(image_ids, num_images, replace = False)
   start = time.time()
   images = list(map(lambda img: single_image_info(img, image_types, paths_list), random_images))
   print(f"Gathered images to display, took {time.time() - start} seconds.")

   # Plot image data.
   fig, axes = plt.subplots(size[0], size[1], figsize = (20, 16))
   for indx, ax in enumerate(axes.flat):
      im = ax.imshow(images[indx], vmin = 0, vmax = 255)
   fig.subplots_adjust(right = 0.8)
   savefig = plt.gcf()
   plt.show()

   # Saves figure to an image file.
   if save:
      if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')):
         os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images'))
      savefig.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'inspected-specific.png'))

def show_random_general_images(num_imgs, image_ids, paths_list, background = "light", save = False):
   """Choose and display all images corresponding to a random selection of image IDs."""
   assert 1 <= num_imgs < 10, f"Number of images should be in range (4, 10), got {num_imgs}."

   # Choose a random selection of images.
   random_images = np.random.choice(image_ids, num_imgs, replace = False)
   start = time.time()
   images = list(map(lambda img: all_image_info(img, paths_list), random_images))
   print(f"Gathered images to display, took {time.time() - start} seconds.")

   # Plot image data.
   order = list(images[0].keys())
   fig, axes = plt.subplots(num_imgs, len(images[0]), figsize = (20, 16))
   if background == "dark":
      fig.patch.set_facecolor('#2e3037ff')
   elif background == 'light':
      fig.patch.set_facecolor('#efefefff')
   gs1 = gspec.GridSpec(num_imgs, len(images[0]))
   gs1.update(wspace = 0.1, hspace = 0.025)
   for indx in range(num_imgs * len(images[0])):
      # Get the ax from the gridspec object.
      ax = plt.subplot(gs1[indx])
      ax.set_aspect('equal')

      image_type = order[indx % len(images[0])]
      if image_type.find('label') != -1:
         image_type_title = image_type[6:]
      else:
         image_type_title = image_type
      if indx // 10 == 0:
         # Replace image title.
         if image_type_title in ['rgb', 'nir']:
            image_type_title = image_type_title.upper()
         else:
            image_type_title = image_type_title.title()

         # Set the title.
         if background == "dark":
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 15, color = 'w')
         else:
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 15, color = 'k')

      # Display the image, with channels reversed.
      ax.imshow(images[indx // 10][image_type], vmin = 0, vmax = 255, cmap = 'magma')

      # Turn off axis.
      ax.axis('off')

   # Display the image.
   savefig = plt.gcf()
   plt.show()

   # Saves figure to an image file.
   if save:
      if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')):
         os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images'))
      savefig.savefig(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images', 'inspected-general.png'))

def show_images_for_id(mode, id, background = 'light'):
   """Displays all images associated with a certain image ID, itself associated with a data mode."""
   global complete_paths
   if mode == 'train':
      paths = train_data_paths
   elif mode == 'val':
      paths = val_data_paths
   elif mode == 'test':
      paths = test_data_paths
   else:
      raise ValueError(f"Received invalid mode {mode}, was expecting train, val, or test.")

   # Get the image paths.
   images = None
   for item in paths:
      if item['id'] == id:
         images = list(map(lambda img: all_image_info(id, complete_paths), id))
   try:
      iter(images)
   except Exception as e:
      raise e

   # Plot image data.
   order = list(images[0].keys())
   fig, axes = plt.subplots(1, len(images[0]), figsize = (20, 16))
   if background == "dark":
      fig.patch.set_facecolor('#2e3037ff')
   elif background == "light":
      fig.patch.set_facecolor('#efefefff')
   gs1 = gspec.GridSpec(1, len(images[0]))
   gs1.update(wspace = 0.1, hspace = 0.025)
   for indx in range(len(images[0])):
      # Get the ax from the gridspec object.
      ax = plt.subplot(gs1[indx])
      ax.set_aspect('equal')

      image_type = order[indx % len(images[0])]
      if image_type.find('label') != -1:
         image_type_title = image_type[6:]
      else:
         image_type_title = image_type
      if indx // 10 == 0:
         # Replace image title.
         if image_type_title in ['rgb', 'nir']:
            image_type_title = image_type_title.upper()
         else:
            image_type_title = image_type_title.title()

         # Set the title.
         if background == "dark":
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 15, color = 'w')
         else:
            ax.set_title(image_type_title.replace('_', ' '),
                         fontsize = 15, color = 'k')

      # Display the image, with channels reversed.
      ax.imshow(images[indx // 10][image_type], vmin = 0, vmax = 255, cmap = 'magma')

      # Turn off axis.
      ax.axis('off')

   # Display the image.
   savefig = plt.gcf()
   plt.show()

if __name__ == '__main__':
   # Construct and parse command line arguments.
   ap = argparse.ArgumentParser()
   ap.add_argument('--directory', default = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Agriculture-Vision'),
                   help = 'The directory path to the dataset, default is ../data/Agriculture-Vision.')
   ap.add_argument('--save', default = False, action = 'store_true', help = 'Pass to save inspected images to a figure image file..')
   ap.add_argument('--mode', default = 'id', help = 'Which mode, either random [general] images, random [specific] images, or an image [id].')
   ap.add_argument('--id', default = None, help = 'If displaying an image ID, then provide the image ID.')
   args = ap.parse_args()

   # Define global values.
   train_data_paths = []; val_data_paths = []; test_data_paths = []; complete_paths = []
   load_json_files(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'Dataset'))

   # Construct a dataset for overwriting.
   dataset = AgricultureVisionDataset()
   dataset.construct()

   # Perform inspection.
   image_files = get_image_names('train', args.directory)
   label_dict = get_classes_dict(args.directory)
   if args.mode == 'specific':
      show_random_specific_images((16, 10), image_files, 'boundary', complete_paths, args.save)
   elif args.mode == 'general':
      show_random_general_images(4, image_files, complete_paths, args.save)
   elif args.mode == 'id':
      show_images_for_id('val', dataset.image_ids['val'][4])
   else:
      raise ValueError("You have provided an invalid mode, should be [random] or [general].")



