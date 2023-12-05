#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J pilot # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 6000 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o pilot_%A.out # Standard out goes to this file 
#SBATCH -e pilot_%A.err # Standard err goes to this filehostname 

source activate merfish
python process_pilot_ingest.py
source deactivate