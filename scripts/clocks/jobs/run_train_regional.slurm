#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J train_region # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 1500 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o train_region_%A.out # Standard out goes to this file 
#SBATCH -e train_region_%A.err # Standard err goes to this filehostname 

source activate merfish
python clocks_regional.py 
source deactivate