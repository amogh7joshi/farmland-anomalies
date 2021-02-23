#!/usr/bin/env python
# -*- coding = utf-8 -*-
import os

import numpy as np
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch

from deeptoolkit.evaluation import concat_training_logs

# Construct the complete training log.
def construct_complete_log():
   """Concatenates all training logs into a single one."""
   # Load each individual training logs.
   log_20 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Dice2D-20.csv'))
   log_40 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Dice-SCL-40.csv'))
   log_60 = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/Dice-SCL-Dice-60.csv'))

   # Concatenate all of them into one, remove an unnecessary column, and return it.
   complete_log = concat_training_logs(log_20, log_40, log_60, save = 'test.csv')
   del complete_log['Unnamed: 0']
   
   # Return the relevant columns.
   return np.array(complete_log.accuracy.tolist()), np.array(complete_log.val_accuracy.tolist()), \
          np.array(complete_log.loss.tolist()), np.array(complete_log.val_loss.tolist()) 

# Create the complete log.
log = construct_complete_log()

# Construct the single-test training log.
def construct_single_log():
   training_log = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/training_log.csv'))
   
   # Convert the dataframe into accuracy/loss arrays.
   accuracy = np.array(training_log.accuracy.tolist())
   val_accuracy = np.array(training_log.val_accuracy.tolist())
   loss = np.array(training_log.loss.tolist())
   val_loss = np.array(training_log.val_loss.tolist())
   
   # Return the final output.
   return accuracy, val_accuracy, loss, val_loss

def plot_single_training_log(accuracy, val_accuracy, loss, val_loss):
   """Plots the training log from a model training session."""
   # Construct the figure.
   plt.figure(figsize = (20, 8))

   # Plot the data.
   plt.plot(np.arange(1, len(accuracy) + 1, 1), accuracy,
            color = 'tab:cyan', label = 'Training Accuracy')
   plt.plot(np.arange(1, len(val_accuracy) + 1, 1), val_accuracy,
            color = 'tab:green', label = 'Validation Accuracy')
   plt.plot(np.arange(1, len(loss) + 1, 1), loss,
            color = 'tab:purple', label = 'Training Loss')
   plt.plot(np.arange(1, len(val_loss) + 1, 1), val_loss,
            color = 'tab:red', label = 'Validation Loss')

   # Change the x-axis and y-axis ticks and labels.
   plt.ylim(min(min(val_accuracy), min(accuracy), min(val_loss) - 0.10, min(loss)),
            min(max(max(val_accuracy), max(accuracy), max(val_loss), max(loss)) + 0.10, 1))
   plt.xlim(1, len(accuracy))
   plt.xticks([i for i in range(0, len(accuracy) + 1, 1)], fontsize = 12)
   plt.xlabel('Epoch', fontsize = 15)

   # Format the graph.
   plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = 0.7)
   plt.axvline(x = 0.1, color = 'black', linewidth = 1.3, alpha = 0.7, zorder = 10)

   # Set the title.
   plt.legend(loc = 'best')

   # Display the plot.
   plt.show()

def plot_complete_log(accuracy, val_accuracy, loss, val_loss, segments = 3):
   """Plots a complete training log with a provided number of segments."""
   # Construct the figure.
   fig, axes = plt.subplots(1, segments, figsize = (20, 8))
   fig.subplots_adjust(wspace = 0)

   # Plot over each ax.
   for indx, ax in enumerate(axes):
      # Get the minimum/maximum  range.
      min_range = int(indx * (len(accuracy) / segments))
      max_range = int((indx + 1) * (len(accuracy) / segments))

      # Plot the data.
      ax.plot(np.arange(min_range, max_range),
              accuracy[min_range: max_range], label = "Accuracy")

      # Set the x/y limits.
      ax.set_ylim([min(accuracy) - 0.1, min(max(accuracy) + 0.1, 1.05)])

      # Remove the y-axis on the middle plots.
      if indx > 0:
         ax.get_yaxis().set_visible(False)

   # Display the plot.
   plt.show()

def plot_only_accuracy(accuracy, val_accuracy, background = 'light'):
   """Plots only model accuracy."""
   # Construct the figure.
   fig, ax = plt.subplots(figsize = (21, 8))

   if background == "dark":
      plt.rcParams['figure.facecolor'] = '#2e3037ff'
   elif background == "light":
      plt.rcParams['figure.facecolor'] = '#efefefff'

   # Plot the data.
   ax.plot(np.arange(1, len(accuracy) + 1, 1), accuracy,
            color = 'tab:cyan', label = 'Training Accuracy')
   ax.plot(np.arange(1, len(val_accuracy) + 1, 1), val_accuracy,
            color = 'tab:green', label = 'Validation Accuracy')

   # Change the x-axis and y-axis ticks and labels.
   ax.set_ylim(min(min(val_accuracy), min(accuracy)) - 0.1,
            min(max(max(val_accuracy), max(accuracy)) + 0.10, 1.0))
   ax.set_xlim(1, len(accuracy))
   plt.xticks([i for i in range(1, len(accuracy) + 1, 1)], fontsize = 12)
   plt.xlabel('Epoch', fontsize = 15)

   # Format the graph.
   ax.axhline(y = plt.axis()[2] + 0.01, color = 'black', linewidth = 1.3, alpha = 0.7)
   ax.axvline(x = 1 + 0.01, color = 'black', linewidth = 1.3, alpha = 0.7, zorder = 10)

   # Split up the graph into sections.
   ax.axvspan(0, 20, color = 'springgreen', alpha = 0.1, zorder = 100)
   ax.axvspan(20, 40, color = 'cornflowerblue', alpha = 0.1, zorder = 100)
   ax.axvspan(40, 60, color = 'blueviolet', alpha = 0.1, zorder = 100)
   legend_items = [
      mpatch.Patch(color = 'springgreen', label = r'$\bf{Stage\;1}:$ Dice Loss'),
      mpatch.Patch(color = 'cornflowerblue', label = r'$\bf{Stage\;2}:$ Surface-Channel Loss'),
      mpatch.Patch(color = 'blueviolet', label = r'$\bf{Stage\;3}:$ Dice Loss'),
   ]

   # Set the legend.
   l1 = plt.legend(loc = 'upper right', facecolor = 'w', framealpha = 1.0)
   plt.legend(handles = legend_items, loc = 'upper left', facecolor = 'w', framealpha = 1.0)
   ax.add_artist(l1)

   # Display the plot.
   plt.show()

if __name__ == '__main__':
   # Construct the log of choice.
   accuracy, val_accuracy, loss, val_loss = construct_complete_log()

   # Construct the plot.
   plot_only_accuracy(accuracy, val_accuracy)
