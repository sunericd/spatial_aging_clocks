#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J anndata # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 120 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=64G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o anndata_%A.out # Standard out goes to this file 
#SBATCH -e anndata_%A.err # Standard err goes to this filehostname 

source activate geneimputation
python scg_create_anndata.py
source deactivate