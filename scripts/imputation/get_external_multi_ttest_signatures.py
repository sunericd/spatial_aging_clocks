'''
Runs TISSUE multiple imputation t-test on different gene signatures

Conda environment used: `requirements/geneimputation.txt`
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from scipy.stats import pearsonr, spearmanr
from sklearn.decomposition import PCA, NMF
import anndata as ad
import warnings
import time
import pickle
import os

from TISSUE.tissue.main import *
from TISSUE.tissue.downstream import *
from TISSUE.tissue.experiments import *
from TISSUE.tissue.utils import *

import logging
logging.getLogger("imported_module").setLevel(logging.WARNING)

import argparse

# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of dataset folder in DataUpload/")
parser.add_argument("genes_to_impute_file", help="name of file in dataset folder containing genes to impute; specify 'all' to impute all available genes")
parser.add_argument("n_imputations", help="number of imputations", type=int)
parser.add_argument("condition", help="condition")
parser.add_argument("group1", help="group1")
parser.add_argument("group2", help="group2")
parser.add_argument("k_gene", help="number of gene groups", type=str)
parser.add_argument("k_cell", help="number of cell groups", type=str)
parser.add_argument("prediction_models", help="prediction model strings separated by '_'", type=str)
parser.add_argument('--symmetric', action='store_true')
parser.add_argument('--non-symmetric', dest='symmetric', action='store_false')
parser.set_defaults(symmetric=True)
parser.add_argument('--preprocess_RNA', action='store_true')
parser.add_argument('--no-preprocess_RNA', dest='preprocess_RNA', action='store_false')
parser.set_defaults(preprocess_RNA=True)
parser.add_argument("sig_dirpath", help="sig_dirpath to save")
args = parser.parse_args()

# load parameters from arguments
dataset_name = args.dataset
genes_to_impute_file = args.genes_to_impute_file
n_imputations = args.n_imputations
condition = args.condition
group1 = args.group1
group2 = args.group2
k_gene = args.k_gene
k_cell = args.k_cell
if k_gene != 'auto':
    k_gene = int(k_gene)
if k_cell != 'auto':
    k_cell = int(k_cell)
symmetric = args.symmetric
preprocess_RNA = args.preprocess_RNA
methods = list(args.prediction_models.split("_"))
sig_dirpath = args.sig_dirpath

if group1.lower() == "none":
    group1 = None
if group2.lower() == "none":
    group2 = None

savedir = "SCPI_k"+str(k_gene)+"_k"+str(k_cell)

# read in spatial and RNAseq datasets
if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
    adata, RNAseq_adata = load_paired_datasets("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                                "DataUpload/"+dataset_name+"/Locations.txt",
                                                "DataUpload/"+dataset_name+"/scRNA_count.txt",
                                                spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
else:
    adata, RNAseq_adata = load_paired_datasets("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                                "DataUpload/"+dataset_name+"/Locations.txt",
                                                "DataUpload/"+dataset_name+"/scRNA_count.txt")
adata.var_names = [x.lower() for x in adata.var_names]
RNAseq_adata.var_names = [x.lower() for x in RNAseq_adata.var_names]


# read in genes to impute list
if genes_to_impute_file == "all":
    target_genes = np.array([x for x in RNAseq_adata.var_names if x not in adata.var_names])
else:
    target_genes = np.genfromtxt(os.path.join("DataUpload/"+dataset_name,genes_to_impute_file), dtype=str)

# preprocess RNAseq data
if preprocess_RNA is True:
    preprocess_data(RNAseq_adata, standardize=False, normalize=True)

# subset spatial data into shared genes
gene_names = np.intersect1d(adata.var_names, RNAseq_adata.var_names)
adata = adata[:, gene_names]

# build spatial graph
build_spatial_graph(adata, method="fixed_radius", n_neighbors=15)


# iterate through different methods and: (1) impute gene expression, (2) compute calibration scores, (3) run DGEA with MI
for method in methods:
    
    # (1) impute gene expression
    if method == "spage":
        if len(adata.var_names) < 40:
            n_pv = 20
        else:
            n_pv = round(np.min([len(adata.var_names), len(adata.obs_names)])/2)
        
        predict_gene_expression (adata, RNAseq_adata, target_genes,
                                 method=method, n_folds=10, n_pv=n_pv)
    elif method == "knn":
        predict_gene_expression (adata, RNAseq_adata, target_genes,
                                 method=method, n_folds=10, n_neighbors=10)
    else:
        predict_gene_expression (adata, RNAseq_adata, target_genes,
                                 method=method, n_folds=10)
    
    # (2) compute calibration scores
    predicted = method+"_predicted_expression"
    calib_genes = [gene for gene in gene_names if gene not in target_genes]
    conformalize_spatial_uncertainty(adata, predicted, calib_genes, weight="exp_cos", mean_normalized=False, add_one=True,
                                     grouping_method="kmeans_gene_cell", k=k_gene, k2=k_cell, n_pc=15, n_pc2=15)
                                     
                   
    # (3) run gene signatures testing with MI
    keys_list = multiple_imputation_gene_signature (sig_dirpath, adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                             group1=group1, group2=group2, symmetric=symmetric, return_keys=True,
                                             load_mi=True)


# Save results
try:
    adata.write(savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}.h5ad")
    # if error loading (i.e. metadata too large), then large_save instead
    try:
        adata2 = sc.read_h5ad(savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}.h5ad")
    except:
        large_save(adata, savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}")
        os.remove(savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}.h5ad")
# if error loading (i.e. metadata too large), then large_save instead
except:
    large_save(adata, savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}")
    os.remove(savedir+"/"+dataset_name+"_"+"_".join(methods)+f"_MI_GENESIG_TTEST_{condition}_{group1}.h5ad")