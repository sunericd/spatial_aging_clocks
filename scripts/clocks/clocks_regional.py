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

#from spatial_propagation import *
from clock_preprocessing import *


os.chdir("/labs/abrunet1/Eric/MERFISH")



restricted_celltype_subset = ['Neuron-Excitatory','Neuron-MSN','Astrocyte','Microglia','Oligodendrocyte','OPC','Endothelial','Pericyte','VSMC','Ependymal','Neuroblast','NSC','Macrophage', 'T cell']


# ## SpatialSmooth (soft pseudobulking)


adata = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")
adata = adata[(adata.obs.clusters!="1")&(adata.obs.mouse_id!="89")&(adata.obs.mouse_id!="67")].copy()


celltypes = pd.unique(adata.obs.celltype).sort_values()
regions = np.unique(adata.obs['region'])
min_cells_for_clock = 100

predicted_ages_all = np.empty(adata.shape[0])

for ct in celltypes:
    for region in regions:
            
        sub_adata = adata[(adata.obs["celltype"]==ct)&(adata.obs["region"]==region),:].copy()
        
        if sub_adata.shape[0] > min_cells_for_clock:
        
            sub_adata = normalize_adata(sub_adata, zscore=False)

            cv_iterator = get_cv_iterator(sub_adata, "mouse_id")

            # cross-validated predictions
            predicted_ages = np.ones(sub_adata.shape[0])*np.nan

            for (train_idxs, test_idxs) in cv_iterator:

                tr_adata = sub_adata[train_idxs,:].copy()
                te_adata = sub_adata[test_idxs,:].copy()

                # Smooth
                tr_adata = spatial_smoothing_expression(tr_adata, alpha=0.8, n_neighbors=5, max_iter=30, tol=1e-2)

                model = LassoCV(cv=5, n_alphas=20, max_iter=10000)
                scaler = StandardScaler()
                pipeline = Pipeline([('transformer', scaler), ('estimator', model)])

                # Train
                pipeline.fit(np.array(tr_adata.X).astype('float64'), tr_adata.obs['age'].values.astype('float64'))

                # Test
                te_adata = spatial_smoothing_expression(te_adata, alpha=0.8, n_neighbors=5, max_iter=30, tol=1e-2)
                pred_ages = pipeline.predict(np.array(te_adata.X).astype('float64'))
                predicted_ages[test_idxs] = pred_ages

            # update adata with predicted age
            predicted_ages_all[(adata.obs["celltype"]==ct)&(adata.obs["region"]==region)] = predicted_ages
    
adata.obs['predicted_age'] = predicted_ages_all


adata.write_h5ad("results/clocks/anndata/REGION_lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh5.h5ad")
