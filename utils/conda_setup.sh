#!/bin/bash

# Download miniconda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh

# Install miniconda
bash Miniconda3-latest-Linux-x86_64.sh

# Follow the instructions:

# Press Enter to read the license.

# Type yes to accept.

# Accept the default installation location.

# When asked whether to initialize Conda, type yes.

source ~/.bashrc

# Create the Conda environment

conda env create -f imgenv.yml

# Activate the environment

conda activate imgenv