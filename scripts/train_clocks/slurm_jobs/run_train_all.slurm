#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J train_all # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 1500 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o train_all_%A.out # Standard out goes to this file 
#SBATCH -e train_all_%A.err # Standard err goes to this filehostname 

source activate merfish
python full_train_clock.py single_cell
python full_train_clock.py pseudobulk_random
python full_train_clock.py spatial_smooth
python full_train_clock.py spatial_smooth_minus80
source deactivate