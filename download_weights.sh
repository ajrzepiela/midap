#!/bin/bash

echo "Download weights"
wget -O model_weights.tar https://polybox.ethz.ch/index.php/s/XoaFLr346h8GxzP/download

echo "Extract model weights"
tar -xvf model_weights.tar

echo "Remove tar-file"
rm model_weights.tar
