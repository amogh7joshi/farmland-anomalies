#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import json

import numpy as np
import matplotlib.pyplot as plt

def get_json_dicts():
   """Loads the JSON dictionaries containing the train/val/test paths and class labels."""
   # Load the dictionaries from their paths.
   with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/Dataset/train.json')) as f:
      train_json_dict = json.load(f)
   with open(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/Dataset/train.json')) as f:
      val_json_dict = json.load(f)

   # Return the dictionaries.
   return train_json_dict, val_json_dict

def get_unique_classes(train, val):
   """Returns the unique classes and counts from the train/val image IDs."""
   # Create a list of classes.
   classes = []

   # Iterate over train/val.
   for dict in [train, val]:
      # Iterate over the items within the dictionary.
      for item in dict:
         # Add the classes to the list.
         classes.extend(item['classes'])

   return np.unique(classes, return_counts = True)

def plot_distribution(unique_classes, class_counts, background = 'light'):
   """Plots the frequency distribution of each class in the dataset."""
   # Prettify the class names.
   unique_classes = [item.replace("_", " ").title() for item in unique_classes]
   class_counts = [item for item in class_counts]

   # Change the order from greatest to least.
   class_counts, unique_classes = zip(*reversed(sorted(zip(class_counts, unique_classes))))

   # Construct the figure.
   fig, ax = plt.subplots(figsize = (8, 8))
   plt.title(r'$\bf{Dataset\;Class\;Imbalance}$', fontsize = 20)
   if background == "dark":
      fig.patch.set_facecolor('#2e3037ff')
   elif background == "light":
      fig.patch.set_facecolor('#efefefff')
   elif background == "white":
      fig.patch.set_facecolor('#ffffff')

   # Plot the distribution.
   ax.bar(np.arange(len(class_counts)), class_counts,
          color = ['cornflowerblue', 'orangered', 'deepskyblue', 'aquamarine', 'darkorange', 'springgreen'])

   # Configure the x- and y-axis.
   plt.xticks(np.arange(len(class_counts)), unique_classes, rotation = 30, fontsize = 14)
   plt.yticks(fontsize = 11)

   # Display the plot.
   plt.show()

def create_pie_distribution(unique_classes, class_counts, background = 'light'):
   """Constructs a pie chart for the dataset class distribution."""
   # Prettify the class names.
   unique_classes = [item.replace("_", " ").title() for item in unique_classes]
   class_counts = [item for item in class_counts]

   # Construct the figure.
   fig, ax = plt.subplots(figsize = (10, 10), constrained_layout = True)
   plt.title(r'$\bf{Dataset\;Class\;Imbalance}$', fontsize = 20)
   if background == "dark":
      fig.patch.set_facecolor('#2e3037ff')
   elif background == "light":
      fig.patch.set_facecolor('#efefefff')
   elif background == "white":
      fig.patch.set_facecolor("#ffffff")

   # Plot the distribution.
   ax.pie(class_counts, labels = unique_classes, shadow = True,
          autopct = '%1.1f%%', startangle = 90, textprops = {'fontsize': 13})
   ax.axis('equal')

   # Configure the rest of the plot.
   ax.legend()

   # Display the plot.
   plt.show()

if __name__ == '__main__':
   # Load the JSON dictionaries.
   train_dict, val_dict = get_json_dicts()

   # Get the unique classes and counts.
   classes, counts = get_unique_classes(train_dict, val_dict)

   # Plot the unique classes frequency distribution.
   create_pie_distribution(classes, counts, background = "white")
