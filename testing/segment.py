#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import random

import cv2
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model

from testing.image import get_testing_image, create_displayable_test_output
from testing.process import postprocess_output
from preprocessing.dataset import AgricultureVisionDataset
from model.loss import dice_loss_2d, surface_channel_loss_2d

# Load the dataset and model.
dataset = AgricultureVisionDataset()
dataset.construct()

model = load_model(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs/save/Model-Dice-SCL-Dice-60.hdf5'),
                   custom_objects = {'dice_loss_2d': dice_loss_2d})

def draw_segmentation_map(main_image, predictions):
   """Draws the segmentation map onto the main image."""
   # Dictionary of colors.
   _COLORS = {5: (0, 38, 255), 2: (0, 128, 255), 3: (50, 168, 82),
              1: (234, 255, 0), 0: (13, 255, 174), 4: (200, 123, 201)}

   # Convert the main image into a usable "display" image.
   if len(main_image) >= 4:
      main_image = np.squeeze(main_image)[:, :, :3]
   else:
      main_image = main_image[:, :, :3]

   main_image = main_image * 255
   main_image = main_image.astype(np.uint8)

   # Iterate over the actual classes.
   classes = [item for item in predictions[1:-1]]
   for level, label in enumerate(classes):
      if not np.any(label):
         # If there is nothing in the label, then there is nothing to do.
         continue

      # Convert the image into a usable "display" image.
      label = np.expand_dims(label, axis = -1)
      label = label * (255 / (np.max(label, axis = (0, 1))))
      label = label.astype(np.uint8)
      _, thresh = cv2.threshold(label, 127, 255, cv2.THRESH_BINARY)

      # Find the contours.
      contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

      # Iterate over the contours and plot them.
      for indx, contour in enumerate(contours):
         # Get the right color for the label.
         color = _COLORS[level]

         # Create another image to draw the mask onto (to generate a filled-in contour).
         overlay = main_image.copy()
         cv2.fillPoly(overlay, pts = [contour], color = color)

         # Convert the label to float32 and bring it to the range of the regular image.
         label = label.astype(np.float32)

         # Draw the filled-in contour onto the main image.
         main_image = cv2.addWeighted(overlay, 0.3, main_image, 0.7, 1.0)

         # Draw a line around the contours to enhance them.
         main_image = cv2.polylines(main_image, pts = [contour], isClosed = True, color = color, thickness = 2)

   # Return the annotated image.
   return main_image

def display_segmented_pair(testing_image, prediction, truth, background = 'light'):
   """Displays a segmented pair of images (the prediction and the ground truth)."""
   # Create the figure.
   fig, axes = plt.subplots(1, 3)
   if background == "dark":
      fig.patch.set_facecolor('#2e3037ff')
   elif background == 'light':
      fig.patch.set_facecolor('#efefefff')

   # Display each of the images on the plots.
   images = [testing_image, truth, prediction]
   for indx, ax in enumerate(axes):
      # Show the image.
      ax.imshow(images[indx])

      # Remove the axes and perform a bit of formatting.
      ax.axis('off')
      if indx == 0:
         ax.set_title("Original Image", fontsize = 15)
      elif indx == 1:
         ax.set_title("Ground Truth", fontsize = 15)
      else:
         ax.set_title("Prediction", fontsize = 15)

   # Display the plot.
   savefig = plt.gcf()
   plt.show()

   # Save the figure.
   savefig.savefig("diagram.png")

if __name__ == '__main__':
   # Load the image data.
   test_image, test_label = get_testing_image('train', 36, with_truth = True)

   # Make predictions on the test image and postprocess the data..
   predicted = model.predict(test_image)
   predicted = postprocess_output(predicted)

   # Convert the test image/label into usable images.
   displayable_test_image = create_displayable_test_output(test_image)
   test_label = postprocess_output(test_label)

   # Draw the contours onto the main image.
   annotated_test_prediction = draw_segmentation_map(displayable_test_image.copy(), predicted)
   annotated_test_truth = draw_segmentation_map(displayable_test_image.copy(), test_label)

   # Display the three images.
   display_segmented_pair(displayable_test_image, annotated_test_prediction, annotated_test_truth)






