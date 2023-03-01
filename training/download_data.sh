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
wget -O midap_training${FILE_EXT} https://polybox.ethz.ch/index.php/s/CQkDDXIKHh68hSf/download

echo
echo "Extracting archive..."
echo "====================="
echo
${EXTRACT} midap_training${FILE_EXT}

echo
echo "Removing archive..."
echo "==================="
echo
rm -vf midap_training${FILE_EXT}
