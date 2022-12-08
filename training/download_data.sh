#!/bin/bash

# Download on Mac
if [[ "$OSTYPE" == "darwin"* ]]; then
  FILE_EXT=".tar"
  EXTRACT="tar -xvf"

# Download on Linux
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
  FILE_EXT=".zip"
  EXTRACT="unzip -o"

# Catch exception
else
  echo "OS type not supported: {$OSTYPE}"
  exit 1
fi

# Actual download
echo "Downloading data..."
echo "==================="
echo
wget -O training_data${FILE_EXT} https://polybox.ethz.ch/index.php/s/WqHSFa17n0aUZK6/download

echo
echo "Extracting archive..."
echo "====================="
echo
${EXTRACT} training_data${FILE_EXT}

echo
echo "Removing archive..."
echo "==================="
echo
rm -vf training_data${FILE_EXT}