#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os
import zipfile

from tensorflow.keras.callbacks import Callback

try:
   from google.colab import files
except ImportError:
   raise ImportError("The SaveModelFiles callback should and can only be used in Google Colab.")

# This file stores a custom callback used in a Google Colaboratory notebook, which
# saves a zip file containing models after a uniform number of epochs, for overnight
# training and even during normal training, as a backup in case something goes wrong.

class SaveModelFiles(Callback):
   def __init__(self, model_dir, frequency = 1):
      """Saves a zipfile containing model files at the completion of an epoch, or
      at a certain frequency, which is determined by the `frequency` argument."""
      # Validate and set the frequency argument.
      if not isinstance(frequency, int):
         raise TypeError(f"The frequency argument should be an integer representing how often "
                         f"model files should be saved, got {type(frequency)}")
      self.frequency = frequency

      # Validate and set the directory from which to search for saved models.
      if not os.path.exists(model_dir):
         raise NotADirectoryError(f"The provided directory {model_dir} does not exist.")
      self.model_dir = model_dir

      # Instantiate the class.
      super(SaveModelFiles, self).__init__()

      # Keep a tracker of the last saved model zipfile, so it can be removed later and clear up
      # disk space in the Colab runtime, especially when they are large models.
      self.previously_saved_file = None

   def on_epoch_end(self, epoch, logs = None):
      """At the end of an epoch, create a zipfile containing the arguments and download it."""
      # If a uniform consistency is provided by which to save the model, only save it those times.
      if not epoch % self.frequency == 0:
         return

      # Remove the tracked file.
      os.remove(self.previously_saved_file)

      # Get the list of only `.hdf5` or `.h5` files in the directory, as they are the only weights files.
      weights_files = os.listdir(self.model_dir)
      weights_files = [os.path.join(self.model_dir, file) for file in weights_files
                       if file.endswith('hdf5') or file.endswith('h5')]

      # Construct the zipfile.
      self.build_archive(epoch, weights_files)

      # Save the zipfile to local machine.
      files.download(f'Model-Archive-{epoch}.zip')

      # Output Message.
      print(f"Successfully downloaded {len(weights_files)} weights files.")

      # Track the current file.
      self.previously_saved_file = os.path.join(self.model_dir, f'Model-Archive-{epoch}.zip')

   @staticmethod
   def build_archive(epoch, weights_files):
      """Constructs the archive zipfile."""
      # Construct zipfile.
      zip_file = zipfile.ZipFile(f'Model-Archive-{epoch}.zip', 'w', zipfile.ZIP_DEFLATED)

      # Write individual paths to the file.
      for wf in weights_files:
         zip_file.write(wf)



