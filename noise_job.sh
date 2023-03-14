#!/bin/bash
#BATCH --partition=short-serial
#SBATCH -o /home/users/tabu/ENSO_clustering/clustering_noise_job-%j.out
#SBATCH -e /home/users/tabu/ENSO_clustering/clusterin_noise_job-%j.err

years=$1

modules
module load jaspy

python /home/users/tabu/ENSO_clustering/"synthetic_classifiability_${years}_optimized_Kaplan.py"


