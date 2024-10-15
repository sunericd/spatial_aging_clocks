'''
Runs cross-validation training/testing of spatial aging clocks on full dataset separated by the two independent cohorts

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
parser.add_argument("batch", help="method for training clock", type=str)
args = parser.parse_args()

method = args.method
batch = args.batch

os.chdir("../..")


# read data and init
adata = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")
adata = adata[adata.obs.batch==batch].copy()

celltypes = np.unique(adata.obs['celltype'])

predicted_ages_all = np.empty(adata.shape[0])


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

    adata.write_h5ad(f"results/clocks/anndata/batch{batch}_lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20.h5ad")