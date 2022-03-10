#!/bin/bash
#SBATCH -c 1
#SBATCH -p cpu


source activate news-tls

python supervised_model.py

