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
sys.path.append("/labs/abrunet1/Eric/MERFISH")

from clock_preprocessing import *
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("method", help="method for training clock", type=str)
args = parser.parse_args()

method = args.method


os.chdir("/labs/abrunet1/Eric/MERFISH")

# read data and init
adata = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")
adata = adata[(adata.obs.clusters!="1")&(adata.obs.mouse_id!="89")&(adata.obs.mouse_id!="67")].copy()

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



# ### Single-cell (with smoothing of predictions)
if method == "single_cell_smooth":

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
            te_adata.obs["predicted_age"] = pred_ages
            
            # Smooth
            pred_ages = spatial_smoothing(te_adata, n_neighbors=10)
            
            predicted_ages[test_idxs] = pred_ages
            
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all


    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_singlecell_smoothed.h5ad")


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


# ### Pseudobulked (random)
if method == "pseudobulk_random_smooth":

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
            te_adata.obs["predicted_age"] = pred_ages
            
            # Smooth
            pred_ages = spatial_smoothing(te_adata, n_neighbors=10)
            
            predicted_ages[test_idxs] = pred_ages
            
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all


    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_B1k_singlecellpred_smoothed.h5ad")



# ### Pseudobulked (spatial)
if method == "pseudobulk_spatial":

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:]
            te_adata = sub_adata[test_idxs,:]
            
            # Pseudobulk
            tr_adata = pseudobulk(tr_adata, ["mouse_id", "celltype"], n=30,
                               obs_to_average=["age", "volume"],
                               obs_to_first=["mouse_id", "celltype"], B=1000, method="spatial")
            
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


    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialPB_singlecellpred.h5ad")


# ### Pseudobulked (spatial)
if method == "pseudobulk_spatial_smooth":

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            
        # cross-validated predictions
        predicted_ages = np.ones(sub_adata.shape[0])*np.nan

        for (train_idxs, test_idxs) in cv_iterator:
            
            tr_adata = sub_adata[train_idxs,:]
            te_adata = sub_adata[test_idxs,:]
            
            # Pseudobulk
            tr_adata = pseudobulk(tr_adata, ["mouse_id", "celltype"], n=30,
                               obs_to_average=["age", "volume"],
                               obs_to_first=["mouse_id", "celltype"], B=1000, method="spatial")
            
            model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
            scaler = StandardScaler()
            pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
            
            # Train
            pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))
            
            # Test
            pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
            te_adata.obs["predicted_age"] = pred_ages
            
            # Smooth
            pred_ages = spatial_smoothing(te_adata, n_neighbors=5)
            
            predicted_ages[test_idxs] = pred_ages
        
        # update adata with predicted age
        predicted_ages_all[adata.obs["celltype"]==ct] = predicted_ages
        
    adata.obs['predicted_age'] = predicted_ages_all


    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialPB_singlecellpred_smoothed.h5ad")




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
if method == "spatial_smooth_noClust1":

    adata = adata[adata.obs.clusters!="1"]

    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

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

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20_noClust1.h5ad")
    
    
    
# ## Soft spatial pseudo-bulking via smoothing of gene expression
if method == "spatial_smooth_noClust1_no89":

    adata = adata[(adata.obs.clusters!="1")&(adata.obs.mouse_id!="89")]

    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

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

    adata.write_h5ad("results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20_noClust1_no89.h5ad")