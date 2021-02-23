#!/usr/bin/env zsh

# A convenience file that hides updates to certain files which I am constantly tweaking.
git update-index --assume-unchanged evaluation/display.py
git update-index --assume-unchanged evaluation/logs.py

if [[ -z $1 ]]; then
  echo "No parameter passed!"
else
  if [ $1 = "hide" ]; then
    git update-index --assume-unchanged evaluation/display.py
    git update-index --assume-unchanged evaluation/logs.py
  elif [ $1 = "show" ]; then
    git update-index --no-assume-unchanged evaluation/display.py
    git update-index --no-assume-unchanged evaluation/logs.py
  fi
fi