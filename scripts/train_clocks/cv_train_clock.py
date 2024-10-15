'''
Runs cross-validation training/testing of spatial aging clocks (and other clocks) on full dataset

Inputs required:
- data/integrated_aging_coronal_celltyped_regioned_raw.h5ad - AnnData object (coronal section dataset)

Conda environment used: `requirements/merfish.txt`
'''

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, lasso_path, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
from matplotlib.collections import PatchCollection
import seaborn as sns

import sys
sys.path.append("/".join(os.getcwd().split("/")[:-2]))

from clock_preprocessing import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("method", help="method for training clock", type=str)
args = parser.parse_args()

method = args.method

os.chdir("../..")


# read data and init
adata = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")

celltypes = np.unique(adata.obs['celltype'])

predicted_ages_all = np.empty(adata.shape[0])



# ### Single-cell
if method == "single_cell":
    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:]
            te_adata = sub_adata[test_idxs,:].copy()
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            predicted_ages[test_idxs] = pred_ages
            
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_singlecell.h5ad")



# ### Pseudobulked (random)
if method == "pseudobulk_random":

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:]
            te_adata = sub_adata[test_idxs,:].copy()
            
            # Pseudobulk
            tr_adata = pseudobulk(tr_adata, ["mouse_id", "celltype"], n=30,
                               obs_to_average=["age", "volume"],
                               obs_to_first=["mouse_id", "celltype"], B=1000)
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            predicted_ages[test_idxs] = pred_ages
            
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all


    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_B1k_singlecellpred.h5ad")



# ## Soft spatial pseudo-bulking via smoothing of gene expression
if method == "spatial_smooth_all":

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:].copy()
            te_adata = sub_adata[test_idxs,:].copy()
            
            # Smooth
            tr_adata = spatial_smoothing_expression(tr_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            te_adata = spatial_smoothing_expression(te_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            predicted_ages[test_idxs] = pred_ages
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20.h5ad")


# ## Soft spatial pseudo-bulking via smoothing of gene expression
if method == "spatial_smooth_onsinglecell":

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:].copy()
            te_adata = sub_adata[test_idxs,:].copy()
            
            # Smooth
            tr_adata = spatial_smoothing_expression(tr_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            predicted_ages[test_idxs] = pred_ages
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsinglecell_alpha08_nneigh20.h5ad")


# spatial smooth (remove higher spillover/misallocation rate genes)
if method == "spatial_smooth_minus80":
    
    exclude_markers = ['Gfap', 'Crym', 'Drd2', 'Nr4a2', 'Ighm', 'Slc17a7', 'Aldoc', 'Adora2a', 'Cd4', 'C1ql3', 'Stmn2', 'Pvalb', 'Thbs4', 'Gja1', 'Atp1a2', 'C4b', 'Drd1', 'Lamp5', 'Slc1a2', 'Sparc', 'Map1lc3a', 'Tox', 'Penk', 'Gad2', 'Chat', 'Apoe', 'Aqp4', 'Sulf2', 'Sox9', 'Clu', 'Tubb3', 'Slc32a1', 'Aldh1l1', 'Spock2', 'Nfic', 'Olig1', 'Flt1', 'Pbx3', 'Pdgfra', 'Adamts3', 'Tac1', 'Cdh2', 'Slc1a3', 'Agpat3', 'Fgfr3', 'Msmo1', 'Ntm', 'Efnb2', 'Apod', 'Cd47', 'Gad1', 'Cdk5r1', 'Cfl1', 'Jak1', 'Sst', 'Sox2', 'Dpp6', 'Stub1', 'Igf2', 'Elovl5', 'Fads2', 'Trim2', 'Syt11', 'C1qa', 'Npy', 'Htt', 'Pcsk1n', 'Akt1', 'Csf1r', 'Igf1r', 'Sox11', 'Slc17a6', 'Mtor', 'C1qb', 'Sod2', 'Btg2', 'Gpm6b', 'Vcam1', 'Nr2e1', 'Parp1']
    adata = adata[:,[gene for gene in adata.var_names if gene not in exclude_markers]]
    print(adata.shape)
    
    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:].copy()
            te_adata = sub_adata[test_idxs,:].copy()
            
            # Smooth
            tr_adata = spatial_smoothing_expression(tr_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            te_adata = spatial_smoothing_expression(te_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            predicted_ages[test_idxs] = pred_ages
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20_minus80.h5ad")