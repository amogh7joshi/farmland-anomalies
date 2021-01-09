#!/usr/bin/env python3

# Generation Paths.
if [ ! -d '../data/Dataset' ]; then
  mkdir ../data/Dataset # Create Path to Output Dataset.
fi

_contains () {  # Check if space-separated list $1 contains line $2
  echo "$1" | tr ' ' '\n' | grep -F -x -q "$2"
}

declare list=(
  'train'
  'val'
  'test'
  'all'
)

# Get generation mode from user.
echo "Choose processing mode: [train|val|test|all]"
read -r MODE
if ! _contains "${list}" "${MODE}"; then
  echo "The mode \"${MODE}\" is not a valid mode. Aborting."
  exit 1
fi

# Determine if generation directory contains files.
if [ ! -z "$(ls -1qA ../data/Dataset)" ]; then
  echo "The Dataset directory already contains files. "
  echo "Overwrite existing files? [y|n]"
  read -r OVERWRITE
  if [ "$OVERWRITE" = "y" ]; then
    rm ../data/Dataset/*.json
  else
    echo "Aborting."
    exit 0
  fi
fi

# Run Generation Script.
cd ..; cd preprocessing || exit 1
python3 generate.py --mode $MODE



