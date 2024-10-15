#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J cv_ssa # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 1500 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o cv_ssa_%A.out # Standard out goes to this file 
#SBATCH -e cv_ssa_%A.err # Standard err goes to this filehostname 

source activate merfish
python cv_train_clock.py spatial_smooth_all
python cv_train_clock.py spatial_smooth_minus80
source deactivate