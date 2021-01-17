#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import pandas as pd
import matplotlib.pyplot as plt

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

def visualize_training_log(training_log) -> None:
   """Visualize training log --> graphs showing model accuracy and loss over epochs."""
   if not os.path.exists(training_log):
      raise FileNotFoundError(f"Training log at {training_log} not found.")
   try:
      log = pd.read_csv(training_log)
   except Exception as e:
      raise e

   # Graph Accuracy.
   plt.xlabel('epoch')
   plt.ylabel('accuracy')
   plt.plot(log['accuracy'], label = 'Training Accuracy')
   plt.plot(log['val_accuracy'], label = 'Validation Accuracy')
   plt.plot(log['loss'], label = 'Training Loss')
   plt.plot(log['val_loss'], label = 'Validation Loss')
   plt.legend(loc = 'upper left')
   plt.show()

