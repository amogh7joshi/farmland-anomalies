#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

from preprocessing.dataset import AgricultureVisionDataset

def view_dataset_types(dataset_obj) -> None:
   """Print type(dataset_obj.<value>_data) for debugging/visualizing."""
   if not isinstance(dataset_obj, AgricultureVisionDataset):
      raise TypeError(f"Invalid object of type {type(dataset_obj)} provided, should be AgricultureVisionDataset.")
   attr_list = []
   if dataset_obj.dtype == 'full':
      attr_list = ['train', 'val', 'test']
   elif dataset_obj.dtype == 'train':
      attr_list = ['train']
   elif dataset_obj.dtype == 'val':
      attr_list = ['val']
   elif dataset_obj.dtype == 'test':
      attr_list = ['test']
   for attr in attr_list:
      print(type(getattr(dataset_obj, f'{attr}_data')))
