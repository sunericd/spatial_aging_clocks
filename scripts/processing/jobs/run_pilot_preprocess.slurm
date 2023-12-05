#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J preprocess # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=64G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o preprocess_%A.out # Standard out goes to this file 
#SBATCH -e preprocess_%A.err # Standard err goes to this filehostname 

source activate merfish
python preprocessing_for_pilot_merfish_data.py
source deactivate