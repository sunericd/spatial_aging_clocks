#!/bin/bash
#SBATCH -n 1 # Number of cores requested 
#SBATCH -J idgeall_nsc # job name
#SBATCH -N 1 # Ensure that all cores are on one machine 
#SBATCH -t 10080 # Runtime in minutes 
#SBATCH -p batch # Partition to submit to 
#SBATCH --account=abrunet1
#SBATCH --mem=512G # Memory per cpu in MB (see also --mem-per-cpu) 
#SBATCH -o idgeall_nsc_%A.out # Standard out goes to this file 
#SBATCH -e idgeall_nsc_%A.err # Standard err goes to this filehostname 

source activate geneimputation
python get_external_multi_ttest.py SVZ all 100 "NSC_neighborhood_type" Strong Weak 4 1 spage_tangram --non-symmetric gene_signatures/NSC_near_vs_far_Sep2023_R2/signatures/
source deactivate