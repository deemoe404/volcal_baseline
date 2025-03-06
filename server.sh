#!/bin/bash
# Initialize conda for the script
eval "$(conda shell.bash hook)"

# Activate conda environment
conda activate ./.conda

# Launch Jupyter Notebook without opening a browser and with no token/password authentication
jupyter notebook --no-browser --NotebookApp.token='' --NotebookApp.password=''
