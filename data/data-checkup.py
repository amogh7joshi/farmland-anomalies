#!/usr/bin/env python3
# -*- coding = utf-8 -*-
import os

# OS functions to look for and fix any issues related to datasets (essentially create folders and such).
# Run from time-to-time to checkup on data.

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Dataset')):
   # If Dataset directory (output path for json files) doesn't exist.
   print("Added Dataset directory for output json files.")
   os.makedirs(os.path.join(os.path.dirname(__file__), 'Dataset'))

if not os.path.exists(os.path.join(os.path.dirname(__file__), 'Agriculture-Vision')):
   # If Agriculture-Vision dataset is missing.
   print("Agriculture-Vision dataset is missing, you need to regenerate it.")

if not os.path.exists(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images')):
   # If the primary output images directory is missing.
   os.makedirs(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'images'))
   print("Added images directory for output images.")

