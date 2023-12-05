#!/bin/bash
#SBATCH -n 4 # Number of cores requested 
#SBATCH -J prepare # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 6000 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=256G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o prepare_%A.out # Standard out goes to this file 
#SBATCH -e prepare_%A.err # Standard err goes to this filehostname 

source activate merfish
python prepare_data_for_clustering.py
source deactivate