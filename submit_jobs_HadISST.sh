#!/bin/bash

job_path='/home/users/tabu/ENSO_clustering'

echo "sending jobs"

for year in 30 40 50 60 70
do
        sbatch -J ${year} /home/users/tabu/ENSO_clustering/noise_job_ERSST.sh ${year}
done


