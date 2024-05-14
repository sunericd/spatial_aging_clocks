'''
Runs cross-validation training/testing of spatial aging clocks in a subregion-specific manner both on the same region and on all external regions

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

#from spatial_propagation import *
from clock_preprocessing import *

os.chdir("../..")



restricted_celltype_subset = ['Neuron-Excitatory','Neuron-MSN','Astrocyte','Microglia','Oligodendrocyte','OPC','Endothelial','Pericyte','VSMC','Ependymal','Neuroblast','NSC','Macrophage', 'T cell']


# ## SpatialSmooth (soft pseudobulking)


adata = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")


celltypes = pd.unique(adata.obs.celltype).sort_values()
regions = np.unique(adata.obs['region'])
min_cells_for_clock = 100



for region in regions:

    predicted_ages_all = np.ones(adata.shape[0])*np.nan

    for ct in celltypes:
        
        ct_adata = adata[(adata.obs["celltype"]==ct)&(adata.obs["region"]!=region),:].copy()
        sub_adata = adata[(adata.obs["celltype"]==ct)&(adata.obs["region"]==region),:].copy()
        
        if sub_adata.shape[0] > min_cells_for_clock:
            
            ct_adata = normalize_adata(ct_adata, zscore=False)
            sub_adata = normalize_adata(sub_adata, zscore=False)

            cv_iterator = get_cv_iterator(sub_adata, "mouse_id")
            cv_iterator_ct = get_cv_iterator(ct_adata, "mouse_id")

            # cross-validated predictions
            predicted_ages = np.ones(ct_adata.shape[0])*np.nan

            for cvi, (train_idxs, test_idxs) in enumerate(cv_iterator):

                tr_adata = sub_adata[train_idxs,:].copy()
                te_adata = ct_adata[cv_iterator_ct[cvi][1],:].copy()

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
                predicted_ages[cv_iterator_ct[cvi][1]] = pred_ages

            # update adata with predicted age
            predicted_ages_all[(adata.obs["celltype"]==ct)&(adata.obs["region"]!=region)] = predicted_ages
    
    
    adata.obs['predicted_age'] = predicted_ages_all

    adata.write_h5ad(f"results/clocks/anndata/RESTRICTED{region.split('/')[0]}_lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh5.h5ad")
