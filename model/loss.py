#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import tensorflow as tf
from tensorflow.keras import backend as K

# Custom loss functions for semantic segmentation.

def dice_loss_2d(true, pred):
   """
   Two-dimensional dice loss for semantic image segmentation, calculating
   the overlap between the true and predicted image.
   """
   # Cast the ground truth to a float.
   true = tf.cast(true, tf.float32)

   # Get the image channels.
   height, width, channels = true.get_shape().as_list()[1:]

   # Flatten the ground truth and predictions.
   pred_flattened = tf.reshape(pred, [-1, height * width * channels])
   true_flattened = tf.reshape(true, [-1, height * width * channels])

   # Find the intersection and denominator term.
   intersection = 2.0 * tf.reduce_sum(pred_flattened * true_flattened, axis = 1) + K.epsilon()
   denominator_term = tf.reduce_sum(pred_flattened, axis = 1) \
                      + tf.reduce_sum(true_flattened, axis = 1) + K.epsilon()

   # Return the final loss term.
   loss = 1 - tf.reduce_mean(intersection / denominator_term)
   return loss

def surface_channel_loss_2d(true, pred):
   """
   Two-dimensional surface-channel loss for semantic image segmentation, calculating
   the maximum error of the difference between true vs. predicted pixels by channel.
   """
   # Cast the ground truth to a float.
   true = tf.cast(true, tf.float32)

   # Get the total squared difference over each channel.
   square_difference = tf.reduce_sum(tf.math.squared_difference(true, pred), axis = [2, 3])

   # Reduce the maximum over the channel axis.
   channel_max = tf.reduce_max(square_difference, axis = -1)

   # Reduce the error over the batch.
   loss = tf.reduce_mean(channel_max, axis = 0)

   # Return the loss.
   return loss

