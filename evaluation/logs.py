#!/usr/bin/env python
# -*- coding = utf-8 -*-
import os

import numpy as np
import pandas as pd
from matplotlib import style
style.use('fivethirtyeight')
import matplotlib.pyplot as plt

# Load the training log.
training_log = pd.read_csv(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/training_log.csv'))

# Convert the dataframe into accuracy/loss arrays.
accuracy = np.array(training_log.accuracy.tolist())
val_accuracy = np.array(training_log.val_accuracy.tolist())
loss = np.array(training_log.loss.tolist())
val_loss = np.array(training_log.val_loss.tolist())

def plot_training_log():
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
   plt.xlim(1, len(accuracy) + 1)
   plt.xticks([i for i in range(0, len(accuracy) + 1, 1)])
   plt.xlabel('Epoch')

   # Format the graph.
   plt.axhline(y = 0, color = 'black', linewidth = 1.3, alpha = 0.7)
   plt.axvline(x = 0.1, color = 'black', linewidth = 1.3, alpha = 0.7, zorder = 10)

   # Set the title.
   plt.legend(loc = 'best')

   # Display the plot.
   plt.show()


plot_training_log()
