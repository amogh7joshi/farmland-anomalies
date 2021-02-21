#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import, division

import os
import sys
import json
import pickle
import argparse

from tqdm import tqdm

import cv2
import numpy as np

from PIL import Image

def get_classes(dataset_dir):
   """Get classes for the Agriculture-Vision dataset."""
   label_dir = os.path.join(dataset_dir, 'train', 'labels')
   return [item for item in os.listdir(label_dir) if item != '.DS_Store']

def get_paths(mode, dataset_dir):
   """Get list of image and directory paths for usage in main generation method."""
   if mode not in ['train', 'val', 'test']:
      raise ValueError("Invalid mode for file and path acquisition: should be train, val, or test.")

   # Create and validate paths to each individual directory.
   image_dir = os.path.join(dataset_dir, mode, 'images')
   assert os.path.exists(image_dir), "Path to dataset feature 'images' missing."
   boundary_dir = os.path.join(dataset_dir, mode, 'boundaries')
   assert os.path.exists(boundary_dir), "Path to dataset feature 'boundaries' missing."
   mask_dir = os.path.join(dataset_dir, mode, 'masks')
   assert os.path.exists(mask_dir), "Path to dataset feature 'masks' missing."
   if mode != 'test':
      label_dir = os.path.join(dataset_dir, mode, 'labels')
      assert os.path.exists(label_dir), "Path to dataset feature 'labels' missing."

   # Create list of filenames for all images.
   rgb_image_dir = os.path.join(image_dir, 'rgb')
   assert os.path.exists(rgb_image_dir), "Path to dataset image feature 'rgb' missing."
   image_files = [name[:-4] for name in os.listdir(rgb_image_dir)]

   # Return filenames and directory paths.
   if mode != 'test':
      return image_dir, boundary_dir, mask_dir, label_dir, image_files
   else:
      return image_dir, boundary_dir, mask_dir, image_files

def generate(mode, output_dir, dataset_dir, generate_class_labels = False):
   """Generate json files containing image path information for use in processing."""
   generation_modes = [] # List containing modes for generation, will be iterated over.
   if mode == 'all':
      generation_modes.extend(['train', 'val', 'test'])
   elif mode == 'train':
      generation_modes.extend('train')
   elif mode == 'val':
      generation_modes.extend('val')
   elif mode == 'test':
      generation_modes.extend('test')
   else:
      raise ValueError("Invalid mode for generation: should be train, val, test, or all.")

   # Iterate over modes and generate json files for each.
   for mode in generation_modes:
      with open(os.path.join(output_dir, f'{mode}.json'), 'w') as file:
         print(f"Generating files for mode {mode}.")
         json_dump = []

         # Get directory paths and images.
         if mode == 'test':
            image_dir, boundary_dir, mask_dir, image_files = get_paths(mode, dataset_dir)
         else:
            image_dir, boundary_dir, mask_dir, label_dir, image_files = get_paths(mode, dataset_dir)

         # Generate actual json files.
         for image_file in tqdm(image_files):
            try:
               image_dict = {
                  'id': image_file,
                  'rgb': os.path.join(image_dir, 'rgb', f'{image_file}.jpg'),
                  'nir': os.path.join(image_dir, 'nir', f'{image_file}.jpg'),
                  'boundary': os.path.join(boundary_dir, f'{image_file}.png'),
                  'mask': os.path.join(mask_dir, f'{image_file}.png')
               }

               # Add label images if mode is train or val, determine image classes from arrays.
               if mode != 'test':
                  data_classes = get_classes(dataset_dir)
                  for data_class in data_classes:
                     image_dict[f'label_{data_class}'] = os.path.join(label_dir, data_class, f'{image_file}.png')

                     # Determine which images demonstrate valid features, add to json.
                     if generate_class_labels:
                        labels, classes = [], []
                        for path_name, class_path in image_dict.items():
                           if 'label' in path_name:
                              class_name = path_name[6:]
                              label_image = np.array(Image.open(os.path.join(dataset_dir, class_path))) / 255
                              if label_image.any():
                                 classes.append(class_name)
                              labels.append(label_image)
                        if ~np.sum(labels, axis = 0).astype(bool).any():
                           classes.append('background')

                        image_dict['classes'] = classes

               # Append image dictionary to complete list of json dumps.
               json_dump.append(image_dict)

            except Exception as e:
               raise e
            finally:
               del image_dict

         # Clean and dump to json file.
         pretty_json_dump = json.dumps(json_dump, indent = 4)
         file.write(pretty_json_dump)

if __name__ == '__main__':
   # Create and parse command line arguments (from script).
   ap = argparse.ArgumentParser()
   ap.add_argument('--root', default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Agriculture-Vision'),
                   help = 'Dataset root directory, default is data/Agriculture-Vision')
   ap.add_argument('--out', default = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Dataset'),
                   help = 'Output directory for json files, default is data/Dataset ')
   ap.add_argument('--mode', default = 'all',
                   help = 'Mode for processing (train|val|test|all), default is all. ')
   args = ap.parse_args()

   # Define dataset path for use in primary methods.
   print("Generating JSON files for Agriculture-Vision Dataset.")
   generate(args.mode, args.out, args.root)
   print("JSON files for Agriculture-Vision Dataset Generated.")



