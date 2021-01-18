#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import sys
from datetime import datetime

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau

from util import view_dataset_types

from preprocessing.dataset import AgricultureVisionDataset
from model.model_factory import load_model

# Construct dataset.
dataset = AgricultureVisionDataset()
dataset.construct()

# Construct Logger.
log_dir = os.path.join(os.path.dirname(__file__), 'logs', 'training-' + datetime.now().strftime("%Y-%m-%d-%H%M"))
if not os.path.exists(log_dir):
   os.makedirs(log_dir)
tensorboard_callback = TensorBoard(log_dir = log_dir, histogram_freq = 1)
save_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models')
if not os.path.exists(save_path): # Create the checkpoint save path.
  os.makedirs(save_path)
ckpt = ModelCheckpoint(os.path.join(save_path, 'Model-{epoch:02d}-{val_accuracy:.4f}.hdf5'),
                       monitor = 'val_loss', verbose = 1, save_best_only = True)
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.1, patience = 8, verbose = 1)

# Construct Model.
model = load_model(dtype = 'light', compile = True)

# Fit model.
results = model.fit(
   dataset.train_data,
   steps_per_epoch = 500,
   epochs = 20,
   validation_data = dataset.val_data,
   validation_steps = 1,
   callbacks = [tensorboard_callback]
)


