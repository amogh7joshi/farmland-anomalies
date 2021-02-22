#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import load_model

from model.loss import dice_loss_2d
from testing.process import preprocess_image, postprocess_output
from preprocessing.dataset import AgricultureVisionDataset

# Construct list of evaluation classes.
evaluation_classes = ['Background', 'Waterway', 'Standing Water', 'Weed Cluster',
                      'Double Plant', 'Planter Skip', 'Cloud Shadow', 'Invalid Pixels']

# Load the model.
model = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data/models/Model-03-0.9039.hdf5'),
                   custom_objects = {'dice_loss_2d': dice_loss_2d})

# Load a testing image.
def get_testing_image(mode, value = None):
   """Load a testing image, either from a provided image path or from a dataset."""
   if mode and value is None:
      # In this case, an image path has been passed.
      if not os.path.exists(mode):
         raise FileNotFoundError("You have only provided an argument for `mode`, this case expects "
                                 "an image filepath. Try passing a different, valid image path.")
      else:
         # Return the processed image.
         return preprocess_image(mode)
   else:
      # Choose the dataset.
      if mode not in ["train", "val", "test", "eval"]:
         raise ValueError(f"Received invalid dataset mode '{mode}', expecting train, val, test, or eval.")
      # Construct the dataset.
      dataset = AgricultureVisionDataset()
      dataset.construct()

      # Get the corresponding dataset.
      if mode == "eval":
         iterator = dataset.evaluation_dataset()
      else:
         iterator = getattr(dataset, f"{mode}_data")

      # Iterate over the data.
      for indx, item in enumerate(iterator):
         # If the index is equal to the value.
         if indx == value:
            if hasattr(item, "numpy"):
               # Item is just a pure piece of image data (from a test set).
               return tf.expand_dims(item.numpy()[0], axis = 0)
            elif len(item) == 2:
               # Item is a train/label data (from train/val).
               return tf.expand_dims(item[0].numpy()[0], axis = 0)
            else:
               # Any other case which has not already been covered.
               return tf.expand_dims(item[0], axis = 0)
         else:
            continue

   # For some reason, if nothing has been returned, then throw an error.
   raise Exception("Nothing was returned, something must be broken.")

def _display_test_image(image):
   """A debugging method to display a test image and shape from an EagerTensor."""
   if hasattr(image, "numpy"):
      print(np.squeeze(image.numpy())[:, :, :3].shape)
      cv2.imshow('val', np.squeeze(image[0].numpy())[:, :, :3])
      cv2.waitKey(0)
   else:
      raise AttributeError("This method is reserved for EagerTensors.")

if __name__ == '__main__':
   # Load the image data.
   test_image = get_testing_image('test', 14)

   # Make predictions on the test image and postprocess it.
   predicted = model.predict(test_image)
   predicted = postprocess_output(predicted)

   # Plot the test image.
   fig, axes = plt.subplots(1, 9, figsize = (20, 6))
   fig.patch.set_facecolor('#2e3037ff')

   # Manually plot the first image.
   axes[0].imshow(cv2.cvtColor(np.squeeze(test_image)[:, :, :3], cv2.COLOR_BGR2RGB))
   axes[0].set_title("Original Image", color = 'w')
   axes[0].axis('off')

   # Plot the remaining images of the predictions.
   for indx, (ax, item) in enumerate(zip(axes[1:], predicted)):
      # Since the item is an EagerTensor, get the array representation.
      if hasattr(item, "numpy"):
         ax.imshow(item.numpy(), cmap = 'gray')
      else:
         ax.imshow(item)

      # Set the title and turn off the axis.
      ax.axis('off')
      ax.set_title(evaluation_classes[indx], color = 'w')

   plt.show()



