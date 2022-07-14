#!/bin/bash

echo "Download weights"
wget -O model_weights.zip https://polybox.ethz.ch/index.php/s/XoaFLr346h8GxzP/download

echo "Download example data"
wget -O example_data.zip https://polybox.ethz.ch/index.php/s/Ub30B0ivoTdGWzK/download

echo "Download psf"
wget -O psf.zip https://polybox.ethz.ch/index.php/s/a1oLGN73UNuxwQv/download

echo "Extract model weights"
unzip model_weights.zip

echo "Extract example data"
unzip example_data.zip

echo "Extract psf"
unzip psf.zip

echo "Remove zip-files"
rm model_weights.zip
rm example_data.zip
rm psf.zip

