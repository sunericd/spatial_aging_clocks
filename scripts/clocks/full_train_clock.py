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



# train clocks


# single cell
if method == "single_cell":

    name = "lasso_cv5_nalphas20_singlecell"

    ################################################
    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        X = np.array(sub_adata.X).astype('float64')
        y = sub_adata.obs['age'].values.astype('float64')
        
        #cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
        model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
        pipeline.fit(X, y)
        
        # save model
        with open(f'results/clocks/{name}_{ct}.pkl', 'wb') as handle:
            pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save training data for downstream purposes
        sub_adata.write_h5ad(f"results/clocks/{name}_{ct}.h5ad")



# pseudobulk random
if method == "pseudobulk_random":
    # Pseudobulk
    adata = pseudobulk(adata, ["mouse_id", "celltype"], n=30,
                       obs_to_average=["age", "volume"],
                       obs_to_first=["mouse_id", "celltype"], B=1000)

    name = "lasso_cv5_nalphas20_randomPB"

    ################################################
    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        X = np.array(sub_adata.X).astype('float64')
        y = sub_adata.obs['age'].values.astype('float64')
        
        #cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
        model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
        pipeline.fit(X, y)
        
        # save model
        with open(f'results/clocks/{name}_{ct}.pkl', 'wb') as handle:
            pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save training data for downstream purposes
        sub_adata.write_h5ad(f"results/clocks/{name}_{ct}.h5ad")


# pseudobulk spatial
if method == "pseudobulk_spatial":

    # Pseudobulk
    adata = pseudobulk(adata, ["mouse_id", "celltype"], n=30,
                       obs_to_average=["age", "volume"],
                       obs_to_first=["mouse_id", "celltype"],
                       obsm_to_first=["spatial"],
                       B=False, method="spatial")

    name = "lasso_cv5_nalphas20_spatialPB"

    ################################################
    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        X = np.array(sub_adata.X).astype('float64')
        y = sub_adata.obs['age'].values.astype('float64')
        
        #cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
        model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
        pipeline.fit(X, y)
        
        # save model
        with open(f'results/clocks/{name}_{ct}.pkl', 'wb') as handle:
            pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save training data for downstream purposes
        sub_adata.write_h5ad(f"results/clocks/{name}_{ct}.h5ad")


# spatial smooth
if method == "spatial_smooth":

    name = "lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20"

    ################################################
    celltypes = np.unique(adata.obs['celltype'])

    predicted_ages_all = np.empty(adata.shape[0])

    for ct in celltypes:
        
        sub_adata = adata[adata.obs["celltype"]==ct,:].copy()
        sub_adata = normalize_adata(sub_adata, zscore=False)
        
        # Smooth
        sub_adata = spatial_smoothing_expression(sub_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
        
        X = np.array(sub_adata.X).astype('float64')
        y = sub_adata.obs['age'].values.astype('float64')
        
        #cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
        model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
        scaler = StandardScaler()
        pipeline = Pipeline([('transformer', scaler), ('estimator', model)])
        pipeline.fit(X, y)
        
        # save model
        with open(f'results/clocks/{name}_{ct}.pkl', 'wb') as handle:
            pickle.dump(pipeline, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        # Save training data for downstream purposes
        sub_adata.write_h5ad(f"results/clocks/{name}_{ct}.h5ad")