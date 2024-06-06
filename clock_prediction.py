'''
Functions for applying spatial aging clocks to make predictions
'''

import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet, ElasticNetCV, Lasso, LassoCV, lasso_path, LinearRegression
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from scipy.stats import pearsonr, spearmanr
import pickle
import os

from clock_preprocessing import *


def get_predictions(adata,
                    clock_obj_name="lasso_cv5_nalphas20_spatialsmooth_alpha08_neigh20",
                    fill_missing="mean",
                    smooth=True,
                    pseudobulk_data=False,
                    normalize=True,
                    standardize=True,
                    scale_ref_data=False,
                    add_in_place=True):
    '''
    Gets predicted ages and adds to adata.obs["predicted_age"] inplace
    
        adata [AnnData] - data for which to obtain predictions for
        clock_obj_name [str] - str identifier for the directory containing cell type-specific pkl model files adn training adata
        fill_missing [str] - how to impute missing gene values for prediction
                           - "mean" --> impute with mean value
                           - "spage" --> uses SpaGE algorithm to impute missing values from training data
        smooth [bool] - whether to smooth ; change to False if no adata.obsm["spatial"] in adata
        pseudobulk_data [bool] - if smooth is False, whether to pseudobulk data instead
        standardize [bool] - whether to standardize data using the pipeline in the pkl files
        scale_ref_data [bool] - whether to scale the reference data before imputation (this should be done if adata.X is scaled)
        add_in_place [bool] - whether to add predictions to adata.obs["predicted_age"] inplace
       
    Returns:
        df [DataFrame] - columns with cell type, pred_age, cohort, and age
    '''
    
    # init for prediction data
    celltype_col = []
    pred_age_col = []
    cohort_col = []
    age_col = []
    id_col = []
    region_col = []
    
    celltypes, counts = np.unique(adata.obs['celltype'], return_counts=True)
    
    # init predicted ages
    predicted_ages = np.ones(len(adata.obs["age"]))*np.nan
    
    # Pseudobulk
    if (smooth is False) and (pseudobulk_data is True):
        adata = pseudobulk(adata, ["mouse_id", "celltype"], n=20,
                           obs_to_first=["mouse_id", "celltype", "cohort", "age"], B=100)
    
    for ct in celltypes:
        
        print(ct)

        if f'{clock_obj_name}_{ct}.pkl' in os.listdir('results/clocks'): # if there exist clock for cell type

            # set up data
            sub_adata = adata[adata.obs["celltype"]==ct,:].copy()

            # load training data to get genes -- normalized and smoothed but not standardized
            tr_adata = sc.read_h5ad(f"results/clocks/{clock_obj_name}_{ct}.h5ad")
            if scale_ref_data is True: # scale training data for imputation
                sc.pp.scale(tr_adata)
            clock_genes = tr_adata.var_names.copy()    

            # subset into shared genes
            intersection = np.intersect1d(clock_genes, sub_adata.var_names)
            sub_adata = sub_adata[:,intersection].copy()
            
            # normalize
            if normalize is True:
                sub_adata = normalize_adata(sub_adata, zscore=False)

            # Smooth
            if smooth is True:
                sub_adata = spatial_smoothing_expression(sub_adata, alpha=0.8, n_neighbors=20, max_iter=30, tol=1e-2)
                        
            # map and impute genes for prediction data
            X = np.ones((sub_adata.shape[0],len(clock_genes)))*np.nan

            missing_genes = []
            missing_idxs = []

            for gidx, gene in enumerate(clock_genes):

                if gene in sub_adata.var_names: # add measured gene
                    X[:,gidx] = np.array(sub_adata[:,gene].X).flatten()

                else: # missing genes
                    missing_genes.append(gene)
                    missing_idxs.append(gidx)
            
            # impute missing genes
            if len(missing_genes) > 0:
                
                print(f"Imputing values for {len(missing_genes)} missing genes")

                if fill_missing == "mean": # basic imputation (i.e. only use shared genes)
                    X[:,missing_idxs] = np.nanmean(tr_adata[:,missing_genes].X, axis=0)

                elif fill_missing == "spage": # SpaGE imputation
                    pb_tr_adata = pseudobulk(tr_adata, ["mouse_id", "celltype"], n=20, B=100, method="random")
                    predicted_expression = spage_impute(sub_adata, pb_tr_adata, missing_genes, n_pv=15)
                    X[:,missing_idxs] = predicted_expression[missing_genes].values
                    X[X<0] = 0 # floor counts

                else:
                    raise Exception("fill_missing not recognized")

            X = X.astype('float64')
            y = sub_adata.obs['cohort'].values.astype('str')
            age = sub_adata.obs['age'].values.astype('str')
            idd = sub_adata.obs['mouse_id'].values.astype('str')
            if "region" in sub_adata.obs.keys():
                region = sub_adata.obs['region'].values.astype('str')
            else:
                region = np.array(['global']*len(idd))

            # load and apply aging clock
            with open(f'results/clocks/{clock_obj_name}_{ct}.pkl', 'rb') as handle:
                pipeline = pickle.load(handle)
            if standardize is True:
                preds = pipeline.predict(X)
            else:
                preds = pipeline[1].predict(X) # skip StandardScaler

            # add results
            celltype_col.append([ct]*len(preds))
            pred_age_col.append(preds)
            cohort_col.append(y)
            age_col.append(age)
            id_col.append(idd)
            region_col.append(region)

            if pseudobulk_data is False:
                predicted_ages[adata.obs["celltype"]==ct] = preds

    # add all predictions and metadata
    celltype_col = np.concatenate(celltype_col)
    pred_age_col = np.concatenate(pred_age_col)
    cohort_col = np.concatenate(cohort_col)
    age_col = np.concatenate(age_col)
    id_col = np.concatenate(id_col)
    region_col = np.concatenate(region_col)
    
    if (add_in_place is True) and (pseudobulk_data is False):
        adata.obs["predicted_age"] = predicted_ages
        
    df = pd.DataFrame(np.vstack((celltype_col,pred_age_col,cohort_col,age_col,id_col,region_col)).T,
                      columns=["celltype", "pred_age", "cohort", "age","mouse_id","region"])
    df['pred_age'] = df['pred_age'].astype(float)
    
    return (df)


def spage_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Runs SpaGE gene imputation
    
    See predict_gene_expression() for details on arguments
    '''
    from SpaGE.main import SpaGE
    
    # transform adata in spage input data format
    if isinstance(spatial_adata.X,np.ndarray):
        spatial_data = pd.DataFrame(spatial_adata.X.T)
    else:
        spatial_data = pd.DataFrame(spatial_adata.X.T.toarray())
    spatial_data.index = spatial_adata.var_names.values
    if isinstance(RNAseq_adata.X,np.ndarray): # convert to array if needed
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T)
    else:
        RNAseq_data = pd.DataFrame(RNAseq_adata.X.T.toarray())
    RNAseq_data.index = RNAseq_adata.var_names.values
    # predict with SpaGE
    predicted_expression = SpaGE(spatial_data.T,RNAseq_data.T,genes_to_predict=genes_to_predict,**kwargs)
    
    return(predicted_expression)