#!/bin/bash
#SBATCH -c 1
#SBATCH -p cpu


source activate news-tls

python train_regression.py

