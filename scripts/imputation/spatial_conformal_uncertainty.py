'''
Makes and saves gene expression predictions made on cross-validation folds.
This script should be run first to generate the intermediate results that are necessary for other analyses.

Example: python spatial_conformal_uncertainty.py Dataset15 10 10 4 1 knn_spage_tangram --save_intermediate --non-symmetric
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

from warnings import simplefilter
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)


import argparse

# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("dataset", help="name of dataset folder in DataUpload/")
parser.add_argument("n_cv_folds", help="number of CV folds to perform", type=int)
parser.add_argument("n_inner_folds", help="either 'all' or an integer for n_folds in predict_gene_expression()", type=str)
parser.add_argument("k_gene", help="number of gene groups", type=str)
parser.add_argument("k_cell", help="number of cell groups", type=str)
parser.add_argument("prediction_models", help="prediction model strings separated by '_'", type=str)
parser.add_argument('--save_intermediate', action='store_true')
parser.add_argument('--dont_save_intermediate', dest='save_intermediate', action='store_false')
parser.set_defaults(save_intermediate=True)
parser.add_argument('--symmetric', action='store_true')
parser.add_argument('--non-symmetric', dest='symmetric', action='store_false')
parser.set_defaults(symmetric=True)
parser.add_argument('--preprocess_RNA', action='store_true')
parser.add_argument('--no-preprocess_RNA', dest='preprocess_RNA', action='store_false')
parser.set_defaults(preprocess_RNA=True)
args = parser.parse_args()

# load parameters from arguments
dataset_name = args.dataset
ncvfolds = args.n_cv_folds
n_folds = args.n_inner_folds
if n_folds == "all":
    n_folds = None
else:
    n_folds = int(n_folds)
k_gene = args.k_gene
k_cell = args.k_cell
if k_gene != 'auto':
    k_gene = int(k_gene)
if k_cell != 'auto':
    k_cell = int(k_cell)
save_intermediate = args.save_intermediate
symmetric = args.symmetric
methods = list(args.prediction_models.split("_"))
preprocess_RNA = args.preprocess_RNA
alpha_levels = np.linspace(0.01, 0.99, 1000)

savedir = "SCPI_k"+str(k_gene)+"_k"+str(k_cell)

# read data
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

# preprocess RNAseq data
if preprocess_RNA is True:
    preprocess_data(RNAseq_adata, standardize=False, normalize=True)

# subset spatial data into shared genes
gene_names = np.intersect1d(adata.var_names, RNAseq_adata.var_names)
adata = adata[:, gene_names]

# folds for CV
np.random.seed(444)
np.random.shuffle(gene_names)
folds = np.array_split(gene_names, ncvfolds)

# run-time results
method_col = []
graph_time_col = []
predict_time_col = []
npredict_col = []
conformalize_time_col = []

# set fold id variable
fold_ids = np.zeros(len(adata.var_names))
for i, fold in enumerate(folds):
    fold_ids[adata.var_names.isin(fold)] = i
adata.var["fold"] = fold_ids.copy()

# make copy
adata_copy = adata.copy()

# results dict
res_dict = {}

# try different methods and make predictions
for method in methods:

    res_dict[method] = {}
    res_dict[method]['ind_gene_results'] = {}
        
    calibration_weight = 0 # for computing weighted average
    test_weight = 0
        
    for i, fold in enumerate(folds):
    
        method_col.append(method)
        
        # subset folds
        sub_adata = adata_copy[:, ~adata_copy.var_names.isin(fold)].copy()
        target_genes = list(fold)
        
        # preprocess spatial data
        preprocess_data(sub_adata, standardize=False, normalize=False)
        
        # Spatial Graph
        start_time = time.time()
        build_spatial_graph(sub_adata, method="fixed_radius", n_neighbors=15)
        graph_time_col.append(time.time() - start_time)
        
        # Predict expression
        start_time = time.time()
        if method == "spage":
            if len(sub_adata.var_names) < 40:
                n_pv = 20
            else:
                n_pv = round(np.min([len(sub_adata.var_names), len(sub_adata.obs_names)])/2)
            
            predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                     method=method, n_folds=n_folds, n_pv=n_pv)
        elif method == "knn":
            predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                     method=method, n_folds=n_folds, n_neighbors=10)
        else:
            predict_gene_expression (sub_adata, RNAseq_adata, target_genes,
                                     method=method, n_folds=n_folds)
        predict_time_col.append(time.time() - start_time)
        npredict_col.append(1+len(sub_adata.var_names))
        
        # Add new predictions
        if i == 0:
            obs_name = method+"_predicted_expression"
            adata.obsm[obs_name] = sub_adata.obsm[obs_name][fold].copy()
        else:
            obs_name = method+"_predicted_expression"
            adata.obsm[obs_name][fold] = sub_adata.obsm[obs_name][fold].copy().values
        
        
        # Spatial Conformal Prediction Intervals    
        
        start_time = time.time()
        
        # get test and calibration genes
        predicted = obs_name
        test_genes = target_genes.copy()
        calib_genes = [gene for gene in gene_names if gene not in test_genes]
        
        # Conformalize and save
        sub_adatac = sub_adata.copy()
        conformalize_spatial_uncertainty(sub_adatac, predicted, calib_genes, weight="exp_cos", mean_normalized=False, add_one=True,
                                         grouping_method="kmeans_gene_cell", k=k_gene, k2=k_cell, n_pc=15)
        
        conformalize_time_col.append(time.time() - start_time)
        
        # save adata within fold
        if save_intermediate is True:
            if not os.path.exists(savedir+"/"+dataset_name+"_intermediate/"):
                os.makedirs(savedir+"/"+dataset_name+"_intermediate/")
            large_save(sub_adatac, savedir+"/"+dataset_name+"_intermediate/"+method+"/"+"fold"+str(i))
            if i == 0: # save folds for downstream work
                np.save(savedir+"/"+dataset_name+"_intermediate/"+method+"/folds.npy", folds)

# save runtimes as dataframe
rt_df = pd.DataFrame([])
rt_df["method"] = method_col
rt_df["predict_time"] = predict_time_col
rt_df["number_of_predicts"] = npredict_col
rt_df["graph_time"] = graph_time_col
rt_df["conformalize_time"] = conformalize_time_col
rt_df.to_csv(savedir+"/"+"runtimes_"+dataset_name+"_"+"_".join(methods)+"_SCPI.csv", index=False)