#!/usr/bin/env bash

if [ -d "../data" ]; then
  # shellcheck disable=SC2164
  cd data
else
  echo "Data directory missing, aborting."
  exit 1
fi

# Determine if dataset has already been created, then
# ask user whether they want to re-preprocess it if so.
if [ -d "../data/Agriculture-Vision" ]; then
  echo "Dataset already exists, do you want to overwrite it? [y|n]"
  read overwrite
  if [ "$overwrite" = "y" ]; then
    rm -rf ../data/Agriculture-Vision
    overwrite=true
  else
    exit 0
  fi
fi

# Create dataset from zip file.
if [ -f "../data/Agriculture-Vision.tar.gz" ]; then
  tar -xvzf ../data/Agriculture-Vision.tar.gz -C ../data/
  sleep 1
  rm -f ../data/Agriculture-Vision/Agriculture-Vision\ Workshop\ Terms\ and\ Conditions.pdf
  echo "Dataset Created."
else
  echo "Compressed file containing dataset is missing, aborting."
fi


