#!/bin/bash

echo "Download weights"
wget -O model_weights.tar https://polybox.ethz.ch/index.php/s/XoaFLr346h8GxzP/download

echo "Download example data"
wget -O example_data.tar https://polybox.ethz.ch/index.php/s/Ub30B0ivoTdGWzK/download

echo "Download psf"
wget -O psf.tar https://polybox.ethz.ch/index.php/s/a1oLGN73UNuxwQv/download

echo "Extract model weights"
tar -xvf model_weights.tar

echo "Extract example data"
tar -xvf example_data.tar

echo "Extract psf"
tar -xvf psf.tar

echo "Remove tar-files"
rm model_weights.tar
rm example_data.tar
rm psf.tar

