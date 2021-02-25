#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib import style
style.use('fivethirtyeight')

from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.metrics import KLDivergence

from prototype import FarmlandAnomalyModel
from preprocessing.dataset import AgricultureVisionDataset

# Beautification dictionary with names.
_METRIC_NICE_NAMES = {'MeanIoU': 'mIoU', 'KLDivergence': 'KL-Divergence'}

def convert_result_to_log(result_values, metrics = None, save = False):
   """Converts the metric results to logs."""
   # Convert items from tensors to numpy arrays.
   result_values = [[item.numpy() for item in metric_list] for metric_list in result_values]

   # Save if required to.
   if save:
      pd.DataFrame(data = result_values, columns = [metric.__class__.__name__ for metric in metrics])\
         .to_csv(os.path.join(os.path.dirname(__file__), 'logs/data.csv'))

   # Convert the items into logs.
   result_values = np.transpose(result_values)

   # Return the items.
   return list(np.array(item) for item in list(result_values))

def preprocess_log(input_log):
   """Preprocesses an inputted log file."""
   # Get the DataFrame columns.
   columns_list = [item for item in input_log.columns if item != "Unnamed: 0"]

   # Get the different metrics and values.
   return_dict = {}
   for column in columns_list:
      # Get the column data.
      return_dict[column] = np.array(input_log[column])

   # Return the data.
   return return_dict

def plot_result_logs(logs):
   """Plots the metric result logs."""
   # Create the figure.
   plt.figure(figsize = (12, 6))

   # Plot the data onto the figure.
   for name, metric_log in logs.items():
      # Get the relevant label.
      try:
         name = _METRIC_NICE_NAMES[name]
      except KeyError:
         # In this case, there is no 'pretty' version of the name,
         # so we simply use the original name.
         name = name

      # Plot the data.
      plt.plot(np.arange(0, len(metric_log)), metric_log, marker = 'o', label = name)
      if name in ['mIoU', 'Precision', 'Recall']:
         plt.text(len(metric_log) - 1, metric_log[-1] - 0.15, f"{metric_log[-1]:.2f}%", ha = 'center')
      else:
         plt.text(len(metric_log) - 1, metric_log[-1] - 0.15, f"{metric_log[-1]:.2f}", ha = 'center')

   # Set the name relevant to each model.
   plt.xticks(np.arange(0, len(metric_log)), ['Stage 1', 'Stage 2', 'Stage 3'], fontsize = 13)

   # Final formatting for the plot.
   plt.ylim(0.2, 2.0)
   plt.legend()

   # Display the plot.
   plt.show()

if __name__ == '__main__':
   # Initialize the different models.
   model_20, model_40, model_60 = FarmlandAnomalyModel.multi_initialize(('first', 'middle', 'final'))

   # Construct the dataset.
   dataset = AgricultureVisionDataset()
   dataset.construct()

   # Create the different metrics.
   miou = MeanIoU(num_classes = 7)
   kld = KLDivergence()

   # Create a tracker list of values.
   model_values = []

   # If there is an existing CSV file containing the metric logs, then
   # load from there (to save time and resources). Only in the other case
   # do we actually run the prediction method to get the results.
   if os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/metric-3stage-log.csv')):
      # Load the log.
      log_result = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/metric-3stage-log.csv'))
      log_result = preprocess_log(log_result)
   else:
      # Otherwise, evaluate the models on the different metrics.
      # Iterate over each model.
      for model in [model_20, model_40, model_60]:
         # Iterate over each piece of training data.
         for x, y in tqdm(iter(dataset.train_data)):
            # Iterate over the different metrics and update each of them.
            for metric in [miou, kld]:
               # Predict the model output.
               output = model.predict(x)

               # Update the states.
               metric.update_state(y, output)

         # Once finished iterating over the training data, create a log from the results.
         results = []
         for metric in [miou, kld]:
            # Update the result of each metric to the list.
            results.append(metric.result())

         # Add the list of results to the main tracker.
         model_values.append(results)
         del results

         # Reset the metric states.
         for metric in [miou, kld]:
            metric.reset_states()

      # Change the values into visible logs.
      log_result = convert_result_to_log(model_values)

   # Plot the log diagrams.
   plot_result_logs(log_result)








