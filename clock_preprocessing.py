'''
Functions for normalizing and preprocessing data for spatial aging clocks
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
from matplotlib.collections import PatchCollection
import seaborn as sns

from spatial_propagation import *


def normalize_adata (adata, zscore=True):
    # NOTE: normalization by volume is already done

    # Normalize total to 250
    sc.pp.normalize_total(adata, target_sum=250)

    # Log transform
    sc.pp.log1p(adata)

    # # Z-score (need to do)
    if zscore is True:
        sc.pp.scale(adata, max_value=10)
        
    return(adata)


def get_cv_iterator(adata, obs_name):
    '''
    Gets an sklearn-compatible CV iterator for unique values of adata.obs[obs_name]
    '''
    cv_iterator = []
    
    n = adata.shape[0]
    
    for i in np.unique(adata.obs[obs_name]):
        trainIndices = (np.arange(n)[adata.obs[obs_name]!=i]).astype(int)
        testIndices =  (np.arange(n)[adata.obs[obs_name]==i]).astype(int)
        cv_iterator.append( (trainIndices, testIndices) )
    
    return(cv_iterator)

# For SingleCell (PB)
def pseudobulk(adata, ident_cols, n=15, obs_to_average=None, obs_to_first=None,
               obsm_to_average=None, obsm_to_first=None, B=False, method="random", random_state=444):
    '''
    Inputs:
        adata [anndata]
            - AnnData object to pseudobulk where rows are single cells
        ident_col [list of str]
            - key in adata.obs where unique values specify pools from which to pseudobulk cells
        n [int]
            - number of single cells to sample from each unique pool to construct each pseudocell
        obs_to_average [None, str or list of str]
            - name of adata.obs columns containing quantities to average for each pseudocell
        obs_to_first [None, str or list of str]
            - name of adata.obsm column containing quantities to take first instance of for each pseudocell
        obsm_to_average [None, str or list of str]
            - name of adata.obs columns containing quantities to average for each pseudocell
        obsm_to_first [None, str or list of str]
            - name of adata.obsm column containing quantities to take first instance of for each pseudocell
        B [int or False]
            - number of total pseudocells created per unique identifier pool
            - can specify False to automatically use the size of the original pool
                - for method=="spatial", this will return pseudocell values in the same order as adata.X
        method [str]
            - "random" for random grouping for pseudobulking
            - "spatial" for spatially nearest neighbors grouping for pseudobulking
    
    Returns:
        pb_adata [anndata]
            - AnnData object where observations are pseudocells
    '''  
    # init groupings based on fused identifiers from ident_cols
    grouping_df = adata.obs.groupby(ident_cols).size().reset_index().rename(columns={0:'count'})
    
    # init objects for pseudobulk results
    pb_X = []
    pb_obs = {}
    pb_obsm = {}
    
    if obs_to_average is not None:
        if isinstance(obs_to_average, str):
            pb_obs[obs_to_average] = []
        else:
            for obs_name in obs_to_average:
                pb_obs[obs_name] = []
    
    if obs_to_first is not None:
        if isinstance(obs_to_first, str):
            pb_obs[obs_to_first] = []
        else:
            for obs_name in obs_to_first:
                pb_obs[obs_name] = []
                
    if obsm_to_average is not None:
        if isinstance(obsm_to_average, str):
            pb_obsm[obsm_to_average] = []
        else:
            for obs_name in obsm_to_average:
                pb_obsm[obs_name] = []
    
    if obsm_to_first is not None:
        if isinstance(obsm_to_first, str):
            pb_obsm[obsm_to_first] = []
        else:
            for obs_name in obsm_to_first:
                pb_obsm[obs_name] = []
    
    # subset into each unique pool and construct pseudocells
    for g in range(grouping_df.shape[0]):
        
        # iterative subsetting
        sub_adata = adata
        for idx in range(len(ident_cols)):
            sub_adata = sub_adata[sub_adata.obs[ident_cols[idx]] == grouping_df[ident_cols[idx]].values[g]]
        
        if sub_adata.shape[0] > 0: # pseudobulk if at least one cell in group
            
            # pseudobulking
            if B is False:
                B_ind = sub_adata.shape[0]
            else:
                B_ind = B

            # determine pseudobulking groups
            if method == "random":
                bootstrap_indices = []
                np.random.seed(random_state)
                random_seeds = np.random.randint(0,1e6,B_ind)
                for b in range(B_ind):
                    np.random.seed(random_seeds[b])
                    bootstrap_indices.append(np.random.choice(np.arange(sub_adata.shape[0]),n))

            elif method == "spatial":
                # compute nearest neighbors
                num_neigh = np.min([n+1, sub_adata.shape[0]-1])
                nbrs = NearestNeighbors(n_neighbors=num_neigh).fit(sub_adata.obsm["spatial"])
                distances_local, indices_local = nbrs.kneighbors(sub_adata.obsm["spatial"])
                # accumulate indices for each cell
                bootstrap_indices = []
                if B is False:
                    cell_center_idxs = np.arange(sub_adata.shape[0])
                else:
                    np.random.seed(random_state)
                    cell_center_idxs = np.random.choice(np.arange(sub_adata.shape[0]),B_ind)
                for cell_idx in cell_center_idxs:
                    bootstrap_indices.append([ni for ni in indices_local[cell_idx,1:num_neigh]])

            for bootstrap_idx in bootstrap_indices:
                # pseudocell expression (average)
                pb_X.append(sub_adata.X[bootstrap_idx,:].mean(axis=0)) # pseudocell

                # pseudocell metadata
                if obs_to_average is not None:
                    if isinstance(obs_to_average, str):
                        obs_to_average = [obs_to_average]
                    for obs_name in obs_to_average:
                        pb_obs[obs_name].append(sub_adata.obs[obs_name].iloc[bootstrap_idx].values.mean())

                if obs_to_first is not None:
                    if isinstance(obs_to_first, str):
                        obs_to_first = [obs_to_first]
                    for obs_name in obs_to_first:
                        pb_obs[obs_name].append(sub_adata.obs[obs_name].iloc[bootstrap_idx].values[0])

                if obsm_to_average is not None:
                    if isinstance(obsm_to_average, str):
                        obsm_to_average = [obsm_to_average]
                    for obs_name in obsm_to_average:
                        pb_obsm[obs_name].append(np.array(sub_adata.obsm[obs_name])[bootstrap_idx,:].mean(axis=0))

                if obsm_to_first is not None:
                    if isinstance(obsm_to_first, str):
                        obsm_to_first = [obsm_to_first]
                    for obs_name in obsm_to_first:
                        pb_obsm[obs_name].append(np.array(sub_adata.obsm[obs_name])[bootstrap_idx[0],:].copy())
                    
    # compile new AnnData object
    pb_X = np.vstack(pb_X)
    pb_adata = ad.AnnData(X=pb_X, dtype="float64")
    
    pb_meta_df = pd.DataFrame.from_dict(pb_obs)
    pb_meta_df.index = pb_adata.obs_names
    pb_adata.obs = pb_meta_df
    
    for key in pb_obsm:
        pb_adata.obsm[key] = np.vstack(pb_obsm[key])
    
    # include original information
    pb_adata.var=adata.var
    pb_adata.varm=adata.varm
    pb_adata.varp=adata.varp
    pb_adata.uns=adata.uns

    return(pb_adata)

# SpatialSmooth
def spatial_smoothing_expression (adata, graph_method="fixed", adjacency_method="binary", group_obs="mouse_id",
                      n_neighbors=15, alpha=0.1, max_iter=100, tol=1e-3, verbose=False):
    '''
    Spatial smoothing for gene expression (i.e. SpatialSmooth)
    
    Refer to spatial_propagation.py for argument descriptions
    '''    
    # init objects for results
    pb_X = adata.X.copy()
    
    # run for each group_obs
    for g in np.unique(adata.obs[group_obs]):
        
        sub_adata = adata[adata.obs[group_obs]==g].copy()
        
        if sub_adata.shape[0] > 1:
            # build graph and adjacency matrix
            build_spatial_graph(sub_adata, method=graph_method, n_neighbors=n_neighbors)
            calc_adjacency_weights(sub_adata, method=adjacency_method)

            X = sub_adata.X.copy()
            S = sub_adata.obsp["S"]

            # propagate predictions
            smoothed_X = propagate (X, S, alpha, max_iter=max_iter, tol=tol, verbose=verbose)
                
            # append results
            pb_X[adata.obs[group_obs]==g, :] = smoothed_X
            
        else:
            pb_X[adata.obs[group_obs]==g, :] = sub_adata.X.copy()
        
    adata.X = pb_X
        
    return(adata)