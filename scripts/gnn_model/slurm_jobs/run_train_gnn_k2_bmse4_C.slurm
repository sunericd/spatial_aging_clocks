#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J k2_bmse4_C # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 3600 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=128G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o k2_bmse4_C_%A.out # Standard out goes to this file 
#SBATCH -e k2_bmse4_C_%A.err # Standard err goes to this filehostname 

source activate merfish_gnn
python train_gnn_model.py 2 2 "T cell,NSC" "celltype" 0.0001 "balanced_mse"
source deactivate