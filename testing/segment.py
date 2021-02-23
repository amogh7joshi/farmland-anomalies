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
   _COLORS = {5: (0, 38, 255), 2: (50, 115, 168), 3: (50, 168, 82),
              1: (234, 255, 0), 0: (13, 255, 174), 4: (200, 123, 201)}

   # Convert the main image into a usable "display" image.
   main_image = np.squeeze(main_image)[:, :, :3]
   main_image = main_image * (255 / (np.max(main_image, axis=(0, 1))))
   main_image = main_image.astype(np.uint8)

   # Iterate over the actual classes.
   classes = [item for item in predictions[1:-1]]
   for level, label in enumerate(classes):
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

         # Draw the filled-in contour onto the main image.
         main_image = cv2.addWeighted(overlay, 0.3, main_image, 0.7, 1.0)

         # Draw a line around the contours to enhance them.
         main_image = cv2.polylines(main_image, pts = [contour], isClosed = True, color = color, thickness = 2)

   # Display the image.
   plt.imshow(main_image)
   plt.axis('off')
   plt.show()

   # Save the image.
   cv2.imwrite('diagram.png', main_image)


if __name__ == '__main__':
   # Load the image data.
   test_image = get_testing_image('eval', 7)

   # Make predictions on the test image and postprocess it.
   predicted = model.predict(test_image)
   predicted = postprocess_output(predicted)

   # Convert the test image into a usable image.
   displayable_test_image = create_displayable_test_output(test_image)

   # Draw the contours onto the main image.
   draw_segmentation_map(test_image, predicted)





