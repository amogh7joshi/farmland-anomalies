#!/usr/bin/env zsh

# OpenCV sometimes encounters issues on MacOS, so this
# script takes care of that issue with the headless distribution.

# Uninstalls existing distribution of OpenCV.
python3 -m pip uninstall opencv-python
python3 -m pip uninstall opencv-python-headless

# Re-install the headless distribution.
python3 -m pip install opencv-python-headless

# Technically unnecessary, but fixed something in the past
# so it's here to stay (upgrades the newly installed distribution).
python3 -m pip install --upgrade opencv-python-headless

