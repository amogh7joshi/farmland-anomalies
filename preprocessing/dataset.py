#!/usr/bin/env python3
# -*- coding = utf-8 -*-
from __future__ import absolute_import, division

import os
import sys
import re
import json
import random

import cv2
import numpy as np

import tensorflow as tf
import tensorflow_datasets as tfds

from PIL import Image

__all__ = ['AgricultureVisionDataset']

np.random.seed(0) # Set random seed to be predictable.

class _AgricultureVisionContainer(object):
   def __init__(self, dtype ='full', augmentation = False, dataset_location = None, processed_paths = None):
      """Agriculture-Vision dataset container class for usage in model training."""
      # Set and parse default arguments.
      assert dtype in ['full', 'train', 'val', 'test']
      self.dtype = dtype
      self.augmentation = augmentation

      # Set default dataset and processed path locations.
      if dataset_location is None: # Set default dataset location.
         self.dataset_location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Agriculture-Vision')
      else:
         assert os.path.exists(dataset_location), f'Cannot find path {dataset_location}'
         self.dataset_location = dataset_location
      if processed_paths is None: # Set default processed paths location.
         self.processed_paths = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Dataset')
      else:
         assert os.path.exists(processed_paths), f'Cannot find path {processed_paths}'
         self.processed_paths = processed_paths

      # Load dictionaries containing image paths and compile image ids.
      self._load_path_data()
      self._set_dataset_type() # Create a list of used datasets, depending on provided datatype.
      self._generate_data_file_list() # Turn path dict into list for future use.
      self._compile_image_ids()

      # Set class label dictionary.
      self._set_label_dict()

   def __len__(self):
      """Return number of image ids in class datasets."""
      return sum(len(data) for data in self.dataset_type_list)

   def __contains__(self, item):
      """Return whether an image id is in the class image id list."""
      return item in self.image_ids

   def __eq__(self, other):
      """Return whether self.dtype == other.dtype (for use in training|testing)."""
      return self.dtype == other.dtype

   def __getitem__(self, item):
      """Get image paths associated with an image ID, which itself is an index of the list self.image_ids."""
      assert isinstance(item, int), f"Invalid type {type(item)}, should be int."
      assert 0 <= int(item) < len(self), f"Index {item} out of range for dataset of type {self.dtype}."

      # Get relevant image path data location from dictionary.
      current_image_id = self.image_ids[item]
      current_image = None; path_subdict = None
      for indx, item in enumerate(self.dataset_type_list):
         if item['id'] == current_image_id:
            current_image = item['id']
            path_subdict = self.dataset_type_list[indx]
            break
      else: # If the value was not found.
         raise KeyError(f"The item {self.image_ids[item]} was not found in the dataset.")

      return path_subdict

   def _load_path_data(self) -> None:
      """Load image path data from processed json files."""
      assert os.path.exists(os.path.join(self.processed_paths, 'train.json'))
      with open(os.path.join(self.processed_paths, 'train.json'), 'r') as train_json_file:
         self.train_data_paths = json.load(train_json_file)
      assert os.path.exists(os.path.join(self.processed_paths, 'val.json'))
      with open(os.path.join(self.processed_paths, 'val.json'), 'r') as val_json_file:
         self.val_data_paths = json.load(val_json_file)
      assert os.path.exists(os.path.join(self.processed_paths, 'test.json'))
      with open(os.path.join(self.processed_paths, 'test.json'), 'r') as test_json_file:
         self.test_data_paths = json.load(test_json_file)

   def _set_dataset_type(self) -> None:
      """Create an internal list of datasets in the class for use in magic methods."""
      type_list = None
      if self.dtype == 'full':
         self.complete_paths = {}
         self.complete_paths.update(self.train_data_paths)
         self.complete_paths.update(self.val_data_paths)
         self.complete_paths.update(self.test_data_paths)
         type_list = self.complete_paths
      if self.dtype == 'train':
         type_list = self.train_data_paths
      if self.dtype == 'val':
         type_list = self.val_data_paths
      if self.dtype == 'test':
         type_list = self.test_data_paths
      self.dataset_type_list = type_list

   def _compile_image_ids(self) -> list:
      """Compile a list of image ids for each one in the dataset."""
      self.image_ids = []
      for img_val in self.dataset_type_list:
         self.image_ids.append(img_val['id'])
      return self.image_ids

   def _generate_data_file_list(self) -> None:
      """Converts the image data dictionary into a list, for use in self.construct()"""
      generated_list = []
      for path_dict in self.dataset_type_list:
         current_list = []
         for key, value in path_dict.items():
            # Skip key == 'id' or 'classes', they are not valid paths.
            if key in ['id', 'classes']: continue
            # Add all files to list.
            current_list.append(value)
         generated_list.append(current_list)
      self.data_file_list = generated_list

   def _set_label_dict(self) -> None:
      """Load a dictionary containing labels for the class."""
      class_dict = {0: 'background'}; count = 1
      label_path = os.path.join(self.dataset_location, 'train', 'labels')
      for item in os.listdir(label_path):
         if item == '.DS_Store' and sys.platform == 'darwin':
            continue  # Skip .DS_Store on MacOS.
         class_dict[count] = str(item)
         count += 1
      self.class_dict = class_dict

   @staticmethod
   def augment(rgb, nir, mask, boundary) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
      """Create a list of spatially augmented images for each one provided (if self.augmented == True)."""
      # Horizontal flip transformation.
      if random.random() > 0.5:
         rgb = tf.image.flip_left_right(rgb)
         mask = tf.image.flip_left_right(mask)
         boundary = tf.image.flip_left_right(boundary)
         nir = tf.image.flip_left_right(nir)

      # Vertical flip transformation
      if random.random() > 0.5:
         rgb = tf.image.flip_up_down(rgb)
         mask = tf.image.flip_up_down(mask)
         boundary = tf.image.flip_up_down(boundary)
         nir = tf.image.flip_up_down(nir)

      # Rotation transformation.
      random_rotation = random.random(); rot_num = None
      if random_rotation < 0.25: rot_num = 1
      if 0.25 > random_rotation > 0.50: rot_num = 2
      if 0.75 > random_rotation > 0.50: rot_num = 3
      rgb = tf.image.rot90(rgb, k = rot_num)
      mask = tf.image.rot90(mask, k = rot_num)
      boundary = tf.image.rot90(boundary, k = rot_num)
      nir = tf.image.rot90(nir, k = rot_num)

      return rgb, mask, boundary, nir

   def image_process(self, paths_list) -> (tf.Tensor, tf.Tensor):
      """Processes an image into corresponding training data and label."""
      # Read images.
      rgb_image = tf.image.decode_image(tf.io.read_file(paths_list[0]), channels = 3)
      nir_image = tf.image.decode_image(tf.io.read_file(paths_list[1]), channels = 1)
      boundary_image = tf.image.decode_image(tf.io.read_file(paths_list[2]), channels = 1)
      mask_image = tf.image.decode_image(tf.io.read_file(paths_list[3]), channels = 1)

      # Concatenate rgb and nir images into a single image.
      nrgb_image = tf.concat([nir_image, rgb_image], axis = 2)
      invalid_pixels = tf.logical_or(boundary_image == 0, mask_image == 0)
      nrgb_image = tf.where(invalid_pixels, tf.zeros_like(nrgb_image), nrgb_image)
      nrgb_image = tf.image.convert_image_dtype(nrgb_image, tf.float32)

      # If test dataset, then no label, just image.
      if self.dtype == 'test':
         nrgb_image.set_shape((512, 512, 4))
         return nrgb_image

      # Create Labels.
      initial_labels = {0: tf.identity(invalid_pixels)}
      for indx, label in self.class_dict.items():
         if indx == 0: continue # Skip index == 0 (background).
         new_indx = indx + 3
         # Read and add image to dict.
         current_label = tf.image.decode_image(tf.io.read_file(paths_list[new_indx]), channels = 1)
         initial_labels[indx] = tf.logical_and(current_label > 0, tf.logical_not(invalid_pixels))
         initial_labels[0] = tf.logical_or(initial_labels[indx], initial_labels[0])

      # Create final image and label arrays.
      initial_labels[0] = tf.logical_not(initial_labels[0])
      final_label = tf.cast(
         tf.concat(
            [initial_labels[0], initial_labels[1], initial_labels[2], initial_labels[3], initial_labels[4],
             initial_labels[5], initial_labels[6], invalid_pixels], axis = 2
         ), dtype = tf.int32
      )

      nrgb_image.set_shape((512, 512, 4))
      final_label.set_shape((512, 512, 8))

      return nrgb_image, final_label

   def create(self):
      """Creates a tf.data.Dataset for dataset type."""
      # Construct list of file locations for all images in class.
      file_locations = self.data_file_list

      # Create datasets.
      dataset = tf.data.Dataset.from_tensor_slices(np.array(file_locations)) \
                               .map(lambda m: self.image_process(m)) \
                               .batch(8).repeat().prefetch(8)
      self.dataset = dataset

      return self.dataset

   def create_evaluation_set(self, batch):
      """Creates a tf.data.Dataset for evaluation."""
      file_locations = self.data_file_list
      if not batch:
         batch = 1

      # Create datasets.
      dataset = tf.data.Dataset.from_tensor_slices(np.array(file_locations)) \
                       .map(lambda m: self.image_process(m)).batch(batch)

      return dataset


class _AgricultureVisionWrapper(_AgricultureVisionContainer):
   def __init__(self, dtype, augmentation = False, dataset_location = None, processed_paths = None):
      """Wrapper class for the Agriculture-Vision Dataset, initializes certain parameters."""
      super(_AgricultureVisionWrapper, self).__init__(dtype, augmentation, dataset_location, processed_paths)
      self.create()

      # Create property for dataset.
      self._create_dynamic_property()

   def _create_dynamic_property(self):
      """Add the class dataset as a property object for easy accessibility."""
      property_name = str(self.dtype + '_data')
      setattr(_AgricultureVisionWrapper, property_name, self.dataset)
   
class AgricultureVisionDataset(object):
   def __init__(self, dtype = 'full', augmentation = None, dataset_location = None, processed_paths = None):
      """
      Accessible class containing the Agriculture-Vision dataset.

      Parameters:

      dtype
            The data type of the class, 'full' by default to return all dataset values, but can be adjusted
            to 'train', 'val', or 'test' if you only want part of the dataset.
      augmentation
            If set to true, this optional parameter allows for image augmentation.
            `NOTE`: This stage has not been implemented yet, to be added in the future.
      dataset_location
            Optional parameter if you have a different dataset location than the existing one.
      processed_paths
            Optional parameter if you have a different processed paths location than the existing one.

      Usage:

      Import and initialize the dataset, then call AgricultureVisionDataset().construct() to initialize
      the dataset parameters. You can either set static variables as the return values of the construct()
      method:

      >>> train_data, val_data, test_data = AgricultureVisionDataset().construct()

      Or, you can use the dynamic class properties `train_data`, `val_data`, and `test_data`.

      >>> dataset = AgricultureVisionDataset()
      >>> dataset.construct()
      >>> print(dataset.train_data)

      """
      # Set and parse default arguments.
      assert dtype in ['full', 'train', 'val', 'test']
      self.dtype = dtype
      # if dtype == 'full': # For convenience in IDEs.
      #    self.train_data = None; self.val_data = None; self.test_data = None
      self.augmentation = augmentation

      # Set default dataset and processed path locations.
      if dataset_location is None: # Set default dataset location.
         self.dataset_location = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Agriculture-Vision')
      else:
         assert os.path.exists(dataset_location), f'Cannot find path {dataset_location}'
         self.dataset_location = dataset_location
      if processed_paths is None: # Set default processed paths location.
         self.processed_paths = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data/Dataset')
      else:
         assert os.path.exists(processed_paths), f'Cannot find path {processed_paths}'
         self.processed_paths = processed_paths

      # Track numpy conversions.
      self._numpy_conversion = False

   def construct(self):
      """Construction method, which develops each dataset."""
      self._numpy_conversion = False
      if self.dtype == 'full':
         complete_data = []
         for indx, dtype in enumerate(['train', 'val', 'test']):
            property_name = str(dtype + '_data')
            cls = _AgricultureVisionContainer(dtype, self.augmentation, self.dataset_location, self.processed_paths)
            cls_dataset = cls.create()
            complete_data.append(cls_dataset)
            setattr(AgricultureVisionDataset, property_name, cls_dataset)
         return complete_data
      else:
         cls = _AgricultureVisionWrapper(self.dtype, self.augmentation, self.dataset_location, self.processed_paths)
         cls_attr = getattr(cls, f'{self.dtype}_data')
         setattr(AgricultureVisionDataset, f'{self.dtype}_data', cls_attr)
         return cls_attr

   def as_numpy(self):
      """Converts datasets to numpy array format, using tensorflow_datasets."""
      if self._numpy_conversion:
         print("Datasets have already been converted to numpy format, skipping conversion.")
         return
      self._numpy_conversion = True
      if self.dtype == 'full':
         if not self.train_data or not self.val_data or not self.test_data:
            raise ValueError("Missing dataset portions, construct dataset first before converting.")
         complete_data = []
         self.train_data = tfds.as_numpy(self.train_data)
         self.val_data = tfds.as_numpy(self.val_data)
         self.test_data = tfds.as_numpy(self.test_data)
         return self.train_data, self.val_data, self.test_data
      else:
         if not hasattr(self, f'{self.dtype}_data'):
            raise ValueError("Missing dataset, construct first before converting.")
         setattr(self, f'{self.dtype}_data', tfds.as_numpy(self.dtype_data))
         return getattr(self, f'{self.dtype}_data')

   @staticmethod
   def evaluation_dataset(batch = None):
      """Returns an evaluation dataset with a batch size of 1."""
      cls = _AgricultureVisionContainer(dtype = 'val')
      return cls.create_evaluation_set(batch)



