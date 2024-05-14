#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J train_batch # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 1500 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o train_batch_%A.out # Standard out goes to this file 
#SBATCH -e train_batch_%A.err # Standard err goes to this filehostname 

source activate merfish
python full_train_clock_batched.py spatial_smooth A
python full_train_clock_batched.py spatial_smooth B
source deactivate