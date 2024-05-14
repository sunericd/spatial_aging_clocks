# Contains main functions for core TISSUE pipeline: computing cell-centric variability and calibrated prediction intervals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import KFold, StratifiedKFold
import anndata as ad
import warnings
import os

#from tissue.utils import nan_weighted_std
#from .utils import nan_weighted_std


def load_paired_datasets (spatial_counts, spatial_loc, RNAseq_counts, spatial_metadata = None,
                          min_cell_prevalence_spatial = 0.0, min_cell_prevalence_RNAseq = 0.01,
                          min_gene_prevalence_spatial = 0.0, min_gene_prevalence_RNAseq = 0.0):
    '''
    Uses datasets in the format specified by Li et al. (2022)
        See: https://drive.google.com/drive/folders/1pHmE9cg_tMcouV1LFJFtbyBJNp7oQo9J
    
    Parameters
    ----------
        spatial_counts [str] - path to spatial counts file; rows are cells
        spatial_loc [str] - path to spatial locations file; rows are cells
        RNAseq_counts [str] - path to RNAseq counts file; rows are genes
        spatial_metadata [None or str] - if not None, then path to spatial metadata file (will be read into spatial_adata.obs)
        min_cell_prevalence_spatial [float between 0 and 1] - minimum prevalence among cells to include gene in spatial anndata object, default=0
        min_cell_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among cells to include gene in RNAseq anndata object, default=0.01
        min_gene_prevalence_spatial [float between 0 and 1] - minimum prevalence among genes to include cell in spatial anndata object, default=0
        min_gene_prevalence_RNAseq [float between 0 and 1] - minimum prevalence among genes to include cell in RNAseq anndata object, default=0
    
    Returns
    -------
        spatial_adata, RNAseq_adata - AnnData objects with counts and location (if applicable) in metadata
    '''
    # Spatial data loading
    spatial_adata = load_spatial_data (spatial_counts,
                                       spatial_loc,
                                       spatial_metadata = spatial_metadata,
                                       min_cell_prevalence_spatial = min_cell_prevalence_spatial,
                                       min_gene_prevalence_spatial = min_gene_prevalence_spatial)
    
    # RNAseq data loading
    RNAseq_adata = load_rnaseq_data (RNAseq_counts,
                                     min_cell_prevalence_RNAseq = min_cell_prevalence_RNAseq,
                                     min_gene_prevalence_RNAseq = min_gene_prevalence_RNAseq)

    return(spatial_adata, RNAseq_adata)


def load_spatial_data (spatial_counts, spatial_loc, spatial_metadata=None,
                       min_cell_prevalence_spatial = 0.0, min_gene_prevalence_spatial = 0.0):
    '''
    Loads in spatial data from text files.
    
    See load_paired_datasets() for details on arguments
    '''
    # read in spatial counts
    df = pd.read_csv(spatial_counts,header=0,sep="\t")
    
    # filter lowly expressed genes
    cells_prevalence = np.mean(df>0, axis=0)
    df = df.loc[:,cells_prevalence > min_cell_prevalence_spatial]
    del cells_prevalence
    
    # filter sparse cells
    genes_prevalence = np.mean(df>0, axis=1)
    df = df.loc[genes_prevalence > min_gene_prevalence_spatial,:]
    del genes_prevalence
    
    # create AnnData
    spatial_adata = ad.AnnData(X=df, dtype='float64')
    spatial_adata.obs_names = df.index.values
    spatial_adata.obs_names = spatial_adata.obs_names.astype(str)
    spatial_adata.var_names = df.columns
    del df
    
    # add spatial locations
    locations = pd.read_csv(spatial_loc,header=0,delim_whitespace=True)
    spatial_adata.obsm["spatial"] = locations.values
    
    # add metadata
    if spatial_metadata is not None:
        metadata_df = pd.read_csv(spatial_metadata)
        metadata_df.index = spatial_adata.obs_names
        spatial_adata.obs = metadata_df
    
    # remove genes with nan values
    spatial_adata = spatial_adata[:,np.isnan(spatial_adata.X).sum(axis=0)==0].copy()
    
    # make unique obs_names and var_names
    spatial_adata.obs_names_make_unique()
    spatial_adata.var_names_make_unique()
    
    return (spatial_adata)


def load_rnaseq_data (RNAseq_counts, min_cell_prevalence_RNAseq = 0.0, min_gene_prevalence_RNAseq = 0.0):
    '''
    Loads in scRNAseq data from text files.
    
    See load_paired_datasets() for details on arguments
    '''
    # read in RNAseq counts
    df = pd.read_csv(RNAseq_counts,header=0,index_col=0,sep="\t")
    
    # filter lowly expressed genes -- note that df is transposed gene x cell
    cells_prevalence = np.mean(df>0, axis=1)
    df = df.loc[cells_prevalence > min_cell_prevalence_RNAseq,:]
    del cells_prevalence
    
    # filter sparse cells
    genes_prevalence = np.mean(df>0, axis=0)
    df = df.loc[:,genes_prevalence > min_gene_prevalence_RNAseq]
    del genes_prevalence
    
    # create AnnData
    RNAseq_adata = ad.AnnData(X=df.T, dtype='float64')
    RNAseq_adata.obs_names = df.T.index.values
    RNAseq_adata.var_names = df.T.columns
    del df
    
    # remove genes with nan values
    RNAseq_adata = RNAseq_adata[:,np.isnan(RNAseq_adata.X).sum(axis=0)==0].copy()
    
    # make unique obs_names and var_names
    RNAseq_adata.obs_names_make_unique()
    RNAseq_adata.var_names_make_unique()
    
    return (RNAseq_adata)



def preprocess_data (adata, standardize=False, normalize=False):
    '''
    Preprocesses adata inplace:
        1. sc.pp.normalize_total() if normalize is True
        2. sc.pp.log1p() if normalize is True
        3. Not recommended: standardize each gene (subtract mean, divide by standard deviation)
    
    Parameters
    ----------
        standardize [Boolean] - whether to standardize genes; default is False
        normalize [Boolean] - whether to normalize data; default is False (based on finding by Li et al., 2022)
    
    Returns
    -------
        Modifies adata in-place
    
    NOTE: Under current default settings for TISSUE, this method does nothing to adata
    '''
    # normalize data
    if normalize is True:
        sc.pp.normalize_total(adata)
        sc.pp.log1p(adata)
    
    # standardize data
    if standardize is True:
        adata.X = np.divide(adata.X - np.mean(adata.X, axis=0), np.std(adata.X, axis=0))


def build_spatial_graph (adata, method="fixed_radius", spatial="spatial", radius=None, n_neighbors=20, set_diag=True):
    '''
    Builds a spatial graph from AnnData according to specifications. Uses Squidpy implementations for building spatial graphs.
    
    Parameters
    ----------
        adata [AnnData] - spatial data, must include adata.obsm[spatial]
        method [str]:
            - "radius" (all cells within radius are neighbors)
            - "delaunay" (triangulation)
            - "delaunay_radius" (triangulation with pruning by max radius; DEFAULT)
            - "fixed" (the k-nearest cells are neighbors determined by n_neighbors)
            - "fixed_radius" (knn by n_neighbors with pruning by max radius)
        spatial [str] - column name for adata.obsm to retrieve spatial coordinates
        radius [None or float/int] - radius around cell centers for which to detect neighbor cells; defaults to Q3+1.5*IQR of delaunay (or fixed for fixed_radius) neighbor distances
        n_neighbors [None or int] - number of neighbors to get for each cell (if method is "fixed" or "fixed_radius" or "radius_fixed"); defaults to 20
        set_diag [True or False] - whether to have diagonal of 1 in adjacency (before normalization); False is identical to theory and True is more robust; defaults to True
    
    Returns
    -------
        Modifies adata in-place
    '''
    # delaunay graph
    if method == "delaunay": # triangulation only
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
    
    # neighborhoods determined by fixed radius
    elif method == "radius":
        if radius is None: # compute 90th percentile of delaunay triangulation as default radius
            sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # build graph
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", set_diag=set_diag)
    
    # delaunay graph with removal of outlier edges with distance > radius
    elif method == "delaunay_radius":
        # build initial graph
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
        if radius is None: # compute default radius as 75th percentile + 1.5*IQR
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # prune edges by radius
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
    
    # fixed neighborhood size with removal of outlier edges with distance > radius
    elif method == "fixed_radius":
        # build initial graph
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
        if radius is None: # compute default radius as 75th percentile + 1.5*IQR
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        # prune edges by radius
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0
            
    # fixed neighborhood size
    elif method == "fixed":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
            
    else:
        raise Exception ("method not recognized")


def load_spatial_graph(adata, npz_filepath, add_identity=True):
    '''
    Reads in scipy sparse adjacency matrix from the specified npz_filepath and adds it to adata.obsp["spatial_connectivities"]
    
    Parameters
    ----------
        add_identity [bool] - whether to add a diagonal of 1's to ensure compatability with TISSUE (i.e. fully connected)
    
    Returns
    -------
        Modifies adata in-place
    
    If graph is weighted, then you should set weight="spatial_connectivities" in downstream TISSUE calls for cell-centric variability calculation
    '''
    from scipy import sparse
    a = sparse.load_npz(npz_filepath)
    
    if add_identity is True:
        a += sparse.identity(a.shape[0]) # add identity matrix

    adata.obsp["spatial_connectivities"] = a
    
    print("If graph is weighted, then you should set weight='spatial_connectivities' in downstream call of conformalize_spatial_uncertainty()")
    

def predict_gene_expression (spatial_adata, RNAseq_adata,
                             target_genes, conf_genes=None,
                             method="spage", n_folds=None, random_seed=444, **kwargs):
    '''
    Leverages one of several methods to predict spatial gene expression from a paired spatial and scRNAseq dataset
    
    Parameters
    ----------
        spatial_adata [AnnData] = spatial data
        RNAseq_adata [AnnData] = RNAseq data, RNAseq_adata.var_names should be superset of spatial_adata.var_names
        target_genes [list of str] = genes to predict spatial expression for; must be a subset of RNAseq_adata.var_names
        conf_genes [list of str] = genes in spatial_adata.var_names to use for confidence measures; Default is to use all genes in spatial_adata.var_names
        method [str] = baseline imputation method
            "knn" (uses average of k-nearest neighbors in RNAseq data on Harmony joint space)
            "spage" (SpaGE imputation by Abdelaal et al., 2020)
            "tangram" (Tangram cell positioning by Biancalani et al., 2021)
            Others TBD
        n_folds [None or int] = number of cv folds to use for conf_genes, cannot exceed number of conf_genes, None is keeping each gene in its own fold
        random_seed [int] = used to see n_folds choice (defaults to 444)
    
    Returns
    -------
        Adds to adata the [numpy matrix]: spatial_adata.obsm["predicted_expression"], spatial_adata.obsm["combined_loo_expression"]
            - matrix of predicted gene expressions (same number of rows as spatial_adata, columns are target_genes)
    '''
    # change all genes to lower
    target_genes = [t.lower() for t in target_genes]
    spatial_adata.var_names = [v.lower() for v in spatial_adata.var_names]
    RNAseq_adata.var_names = [v.lower() for v in RNAseq_adata.var_names]
    
    # drop duplicates if any (happens in Dataset14)
    if RNAseq_adata.var_names.duplicated().sum() > 0:
        RNAseq_adata = RNAseq_adata[:,~RNAseq_adata.var_names.duplicated()].copy()
    if spatial_adata.var_names.duplicated().sum() > 0:
        spatial_adata = spatial_adata[:,~spatial_adata.var_names.duplicated()].copy()
    
    # raise warning if any target_genes in spatial data already
    if any(x in target_genes for x in spatial_adata.var_names):
        warnings.warn("Some target_genes are already measured in the spatial_adata object!")
    
    # First pass over all genes using specified method
    if method == "knn":
        predicted_expression_target = knn_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "spage":
        predicted_expression_target = spage_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "gimvi":
        predicted_expression_target = gimvi_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    elif method == "tangram":
        predicted_expression_target = tangram_impute(spatial_adata,RNAseq_adata,genes_to_predict=target_genes,**kwargs)
    else:
        raise Exception ("method not recognized")
        
    # Second pass over conf_genes using specified method using cross-validation
    
    if conf_genes is None:
        conf_genes = list(spatial_adata.var_names)
    conf_genes = [c.lower() for c in conf_genes]
    conf_genes_unique = [c for c in conf_genes if c not in target_genes] # removes any conf_genes also in target_genes
    if len(conf_genes_unique) < len(conf_genes):
        print("Found "+str(len(conf_genes)-len(conf_genes_unique))+" duplicate conf_gene in target_genes.")
    conf_genes_RNA = [c for c in conf_genes_unique if c in RNAseq_adata.var_names] # remove any conf genes not in RNAseq
    if len(conf_genes_RNA) < len(conf_genes_unique):
        print("Found "+str(len(conf_genes_unique)-len(conf_genes_RNA))+" conf_gene not in RNAseq_adata.")
    conf_genes = conf_genes_RNA
    
    # raise error if no conf_genes
    if len(conf_genes) == 0:
        raise Exception ("No suitable conf_genes specified!")
    
    # create folds if needed
    if n_folds is None:
        n_folds = len(conf_genes)
    elif n_folds > len(conf_genes):
        raise Warning ("n_folds in predict_gene_expression() is greater than length of conf_genes...")
        n_folds = len(conf_genes)

    np.random.seed(random_seed)
    np.random.shuffle(conf_genes)
    folds = np.array_split(conf_genes, n_folds)
    
    # run prediction on each fold
    for gi, fold in enumerate(folds):
        if method == "knn":
            loo_expression = knn_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "spage":
            loo_expression = spage_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "gimvi":
            loo_expression = gimvi_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        elif method == "tangram":
            loo_expression = tangram_impute(spatial_adata[:,~spatial_adata.var_names.isin(fold)],RNAseq_adata,genes_to_predict=list(fold)+target_genes,**kwargs)
        else:
            raise Exception ("method not recognized")
    
        # Update 
        if gi == 0:
            predicted_expression_conf = loo_expression.copy()
        else:
            predicted_expression_conf['index'] = range(predicted_expression_conf.shape[0])
            loo_expression['index'] = range(loo_expression.shape[0])
            predicted_expression_conf.set_index('index')
            loo_expression.set_index('index')
            predicted_expression_conf = pd.concat((predicted_expression_conf,loo_expression)).groupby(by="index").sum().reset_index().drop(columns=['index'])
    
    # Take average of target_genes (later overwritten by "all genes"-predicted)
    predicted_expression_conf[target_genes] = predicted_expression_conf[target_genes]/(len(conf_genes))
    
    # Update spatial_adata
    predicted_expression_target.index = spatial_adata.obs_names
    predicted_expression_conf.index = spatial_adata.obs_names

    # gets predictions for target genes followed by conf genes
    predicted_expression_target[conf_genes] = predicted_expression_conf[conf_genes].copy()
    spatial_adata.obsm[method+"_predicted_expression"] = predicted_expression_target
    
    spatial_adata.uns["conf_genes_used"] = conf_genes
    spatial_adata.uns["target_genes_used"] = target_genes


def knn_impute (spatial_adata, RNAseq_adata, genes_to_predict, n_neighbors, **kwargs):
    '''
    Runs basic kNN imputation using Harmony subspace
    
    See predict_gene_expression() for details on arguments
    '''
    from scanpy.external.pp import harmony_integrate
    from scipy.spatial.distance import cdist
    
    # combine anndatas
    intersection = np.intersect1d(spatial_adata.var_names, RNAseq_adata.var_names)
    subRNA = RNAseq_adata[:, intersection]
    subspatial = spatial_adata[:, intersection]
    joint_adata = ad.AnnData(X=np.vstack((subRNA.X,subspatial.X)), dtype='float32')
    joint_adata.obs_names = np.concatenate((subRNA.obs_names.values,subspatial.obs_names.values))
    joint_adata.var_names = subspatial.var_names.values
    joint_adata.obs["batch"] = ["rna"]*len(subRNA.obs_names.values)+["spatial"]*len(spatial_adata.obs_names.values)
    
    # run Harmony
    sc.tl.pca(joint_adata)
    harmony_integrate(joint_adata, 'batch', verbose=False)
    
    # kNN imputation
    knn_mat = cdist(joint_adata[joint_adata.obs["batch"] == "spatial"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])],
                     joint_adata[joint_adata.obs["batch"] == "rna"].obsm['X_pca_harmony'][:,:np.min([30,joint_adata.obsm['X_pca_harmony'].shape[1]])])
    k_dist_threshold = np.sort(knn_mat)[:, n_neighbors-1]
    knn_mat[knn_mat > k_dist_threshold[:,np.newaxis]] = 0 # sets all dist > thresh to 0
    knn_mat[knn_mat > 0] = 1 # 1 for connection to a nn
    row_sums = knn_mat.sum(axis=1)
    knn_mat = knn_mat / row_sums[:,np.newaxis]
    predicted_expression = knn_mat @ RNAseq_adata.X
    
    predicted_expression = pd.DataFrame(predicted_expression, columns=RNAseq_adata.var_names.values)
    predicted_expression = predicted_expression[genes_to_predict]
    
    return(predicted_expression)
    
    
def spage_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Runs SpaGE gene imputation
    
    See predict_gene_expression() for details on arguments
    '''
    #from tissue.SpaGE.main import SpaGE
    from .SpaGE.main import SpaGE
    
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


def tangram_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Run Tangram gene imputation (positioning) using the more efficient cluster-level approach with Leiden clustering
    
    See predict_gene_expression() for details on arguments
    '''
    import torch
    from torch.nn.functional import softmax, cosine_similarity, sigmoid
    import tangram as tg
    
    # clustering and preprocessing
    RNAseq_adata_label = RNAseq_adata.copy()
    sc.pp.highly_variable_genes(RNAseq_adata_label)
    RNAseq_adata_label = RNAseq_adata[:, RNAseq_adata_label.var.highly_variable].copy()
    sc.pp.scale(RNAseq_adata_label, max_value=10)
    sc.tl.pca(RNAseq_adata_label)
    sc.pp.neighbors(RNAseq_adata_label)
    sc.tl.leiden(RNAseq_adata_label, resolution = 0.5)
    RNAseq_adata.obs['leiden'] = RNAseq_adata_label.obs.leiden
    del RNAseq_adata_label
    tg.pp_adatas(RNAseq_adata, spatial_adata) # genes=None default using all genes shared between two data
    
    # gene projection onto spatial
    ad_map = tg.map_cells_to_space(RNAseq_adata, spatial_adata, mode='clusters', cluster_label='leiden', density_prior='rna_count_based', verbose=False)
    ad_ge = tg.project_genes(ad_map, RNAseq_adata, cluster_label='leiden')
    predicted_expression = pd.DataFrame(ad_ge[:,genes_to_predict].X, index=ad_ge[:,genes_to_predict].obs_names, columns=ad_ge[:,genes_to_predict].var_names)
    
    return(predicted_expression)


def gimvi_impute (spatial_adata, RNAseq_adata, genes_to_predict, **kwargs):
    '''
    Run gimVI gene imputation
    
    See predict_gene_expression() for details on arguments
    '''
    import scvi
    from scvi.external import GIMVI
    
    # preprocessing of data
    spatial_adata = spatial_adata[:, spatial_adata.var_names.isin(RNAseq_adata.var_names)]
    predict_idxs = [list(RNAseq_adata.var_names).index(gene) for gene in genes_to_predict]
    spatial_dim0 = spatial_adata.shape[0]
    
    # indices for filtering out zero-expression cells
    filtered_cells_spatial = (spatial_adata.X.sum(axis=1) > 1)
    filtered_cells_RNAseq = (RNAseq_adata.X.sum(axis=1) > 1)
    
    # make copies of subsets
    spatial_adata = spatial_adata[filtered_cells_spatial,:].copy()
    RNAseq_adata = RNAseq_adata[filtered_cells_RNAseq,:].copy()
    
    # setup anndata for scvi
    GIMVI.setup_anndata(spatial_adata)
    GIMVI.setup_anndata(RNAseq_adata)
    
    # train gimVI model
    model = GIMVI(RNAseq_adata, spatial_adata, **kwargs)
    model.train(200)
    
    # apply trained model for imputation
    _, imputation = model.get_imputed_values(normalized=False)
    imputed = imputation[:, predict_idxs]
    predicted_expression = np.zeros((spatial_dim0, imputed.shape[1]))
    predicted_expression[filtered_cells_spatial,:] = imputed
    predicted_expression = pd.DataFrame(predicted_expression, columns=genes_to_predict)
    
    return(predicted_expression)

    
def conformalize_spatial_uncertainty (adata, predicted, calib_genes, weight='uniform', mean_normalized=False, add_one=True,
                                      grouping_method=None, k='auto', k2='auto', n_pc=None, n_pc2=None):
    '''
    Generates cell-centric variability and then performs stratified grouping and conformal score calculation
    
    Parameters
    ----------
        adata - AnnData object with adata.obsm[predicted] and adata.obsp['spatial_connectivites']
        predicted [str] - string corresponding to key in adata.obsm that contains the predicted transcript expression
        calib_genes [list or np.1darray] - strings corresponding to the genes to use in calibration
        weight [str] - weights to use when computing spatial variability (either 'exp_cos' or 'uniform'; default is 'uniform')
        mean_normalized [bool] - whether the standard deviation will be mean-normalized (i.e. coefficient of variation)
        add_one [bool] - whether to add an intercept term of one to the spatial standard deviation
        For grouping_method [str], k [int>0 or 'auto'], k2 [None or int>0 or 'auto'], n_pc [None or int>0], n_pc2 [None or int>0]; refer to get_grouping()
    
    Returns
    -------
        Saves the uncertainty in adata.obsm[predicted+"_uncertainty"]
        Saves the scores in adata.obsm[predicted+"_score"]
        Saves an upper and lower bound in adata.obsm[predicted+"_lo"/"_hi"]
    '''
    # get spatial uncertainty and add to annotations
    scores, residuals, G_stdev, G = get_spatial_uncertainty_scores(adata, predicted, calib_genes,
                                                                   weight=weight,
                                                                   mean_normalized=mean_normalized,
                                                                   add_one=add_one)
    
    adata.obsm[predicted+"_uncertainty"] = pd.DataFrame(G_stdev,
                                                        columns=adata.obsm[predicted].columns,
                                                        index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_score"] = pd.DataFrame(scores,
                                                  columns=calib_genes,
                                                  index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_error"] = pd.DataFrame(residuals,
                                                  columns=calib_genes,
                                                  index=adata.obsm[predicted].index)                                              
        
    # define group
    if grouping_method is None:
        groups = np.zeros(G.shape)
    else:
        groups, k_final, k2_final = get_grouping(G, method=grouping_method, k=k, k2=k2, n_pc=n_pc, n_pc2=n_pc2)
    
    # add grouping and k-values to anndata
    adata.obsm[predicted+"_groups"] = groups
    adata.uns[predicted+"_kg"] = k_final
    adata.uns[predicted+"_kc"] = k2_final
    
    
def get_spatial_uncertainty_scores (adata, predicted, calib_genes, weight='uniform', mean_normalized=False,
                                    add_one=True):
    '''
    Computes spatial uncertainty scores (i.e. cell-centric variability)
    
    Parameters
    ----------
        adata - AnnData object with adata.obsm[predicted] and adata.obsp['spatial_connectivites']
        predicted [str] - string corresponding to key in adata.obsm that contains the predicted transcript expression
        calib_genes [list or np.1darray] - strings corresponding to the genes to use in calibration
        weight [str] - weights to use when computing spatial variability (either 'exp_cos', 'uniform', or 'spatial_connectivities')
                     - 'spatial_connectivities' will use values in adata.obsp['spatial_connectivities']
        mean_normalized [bool] - whether the standard deviation will be mean-normalized (i.e. coefficient of variation)
        add_one [bool] - whether to add one to the uncertainty
        
    Returns
    -------
        scores - spatial uncertainty scores for all calib_genes
        residuals - prediction errors matching scores dimensions
        G_stdev - spatial standard deviations measured; same shape as adata.obsm[predicted]
        G - adata.obsm[predicted].values
    '''
    if weight not in ["uniform", "exp_cos", "spatial_connectivities"]:
        raise Exception('weight not recognized')
    
    if 'spatial_connectivities' not in adata.obsp.keys():
        raise Exception ("'spatial_connectivities' not found in adata.obsp and is required")
        
    # compute standard deviations
    if isinstance(adata.obsp["spatial_connectivities"],np.ndarray):
        A = adata.obsp["spatial_connectivities"].copy() # has 1 on diagonals
    else:
        A = adata.obsp["spatial_connectivities"].toarray().copy()
    
    A[A == 0] = np.nan # convert zero to nan
    G = adata.obsm[predicted].values.copy()
    G_stdev = np.zeros_like(G)
    
    # compute weights
    if weight == "exp_cos": # use TISSUE cosine similarity weighting
        from sklearn.metrics.pairwise import cosine_similarity
        cos_weights = cosine_similarity(G)
        weights = np.exp(cos_weights)
    elif weight == "spatial_connectivities": # use preset weights
        weights = A.copy()
        weights[np.isnan(weights)] = 0
        A[A>0] = 1
    else:
        weights = np.ones(A.shape)
    
    for j in range(G.shape[1]):
        # multiply to get neighbors (row-wise element-wise multiplication)
        nA = A*G[:,j]
        
        # compute variability
        nA_std = []
        for i in range(nA.shape[0]):
            nA_std.append(cell_centered_variability(nA[i,:], weights=weights[i,:], c_idx=i))
        nA_std = np.array(nA_std)
        
        # mean normalization if specified
        if mean_normalized is True: # coefficient of variation (SD/mu) -- masked for SD > 0
            nA_std[nA_std>0] = nA_std[nA_std>0] / np.nanmean(nA, axis=1)[nA_std>0]
        
        # add one if specified
        if add_one is True:
            nA_std += 1
        
        # update G_stdev with uncertainties
        G_stdev[:,j] = nA_std
    
    # compute scores based on confidence genes (prediction residuals)
    calib_idxs = [np.where(adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]
    residuals = adata[:, calib_genes].X - adata.obsm[predicted][calib_genes].values # Y-G
    
    warnings.filterwarnings("ignore", category=RuntimeWarning) # suppress RuntimeWarning for division by zero
    scores = np.abs(residuals) / G_stdev[:, calib_idxs] # scores
    warnings.filterwarnings("default", category=RuntimeWarning)
    
    return(scores, residuals, G_stdev, G)


def cell_centered_variability (values, weights, c_idx):
    '''
    Takes in an array and weights to compute cell-centric variability:
    
    Parameters
    ----------
        values [1d arr] - array with cell's masked neighborhood expression (non-neighbors are nan)
        weights [1d arr] - same dim as values; contains weights for computing CCV_c
        c_idx [int] - index for which element of nA corresponds to center cell
        
    Returns
    -------
        ccv [float] - cell-centric varaiblity
    '''
    values_f = values[np.isfinite(values)]
    weights_f = weights[np.isfinite(values)]
    average = values[c_idx] # "average" is simply the center cell value
    variance = np.average((values_f-average)**2, weights=weights_f)
    ccv = np.sqrt(variance)
    
    return(ccv)


def get_spatial_uncertainty_scores_from_metadata(adata, predicted):
    '''
    Returns scores, residuals, G_stdev, G (outputs of get_spatial_uncertainty_scores) from precomputed entries
    in the AnnData (adata) object. Note, these must have been computed and saved in the same was as in
    conformalize_spatial_uncertainty().
    
    Parameters
    ----------
        adata [AnnData] - object that has saved results in obsm
        predicted [str] - key for predictions in obsm
        
    Returns
    -------
        scores - array of calibration scores [cell x gene]
        residuals - prediction error [cell x gene]
        G_stdev - array of cell-centric variability measures [cell x gene]
        groups - array of indices for group assignment [cell x gene]
    '''
    scores = np.array(adata.obsm[predicted+"_score"]).copy()
    residuals = np.array(adata.obsm[predicted+"_error"]).copy()
    G_stdev = np.array(adata.obsm[predicted+"_uncertainty"]).copy()
    G = np.array(adata.obsm[predicted]).copy()
    groups = np.array(adata.obsm[predicted+"_groups"]).copy()
    
    return(scores, residuals, G_stdev, G, groups)


def get_grouping(G, method, k='auto', k2='auto', min_samples=5, n_pc=None, n_pc2=None):
    '''
    Given the predicted gene expression matrix G (rows=cells, cols=genes),
    creates a grouping of the different genes (or cells) determined by:
    
    Parameters
    ----------
        G [numpy matrix/array] - predicted gene expression; columns are genes
        method [str] - 'cv_exp' to separate by quantiles of CV in gene expression
                       'kmeans_gene' to separate genes by k-means clustering
                       'kmeans_cell' to separate cells by k-means clustering
                       'kmeans_gene_cell' to separate by genes and the by cells by k-means clustering
        k [int] - number of groups; only for cv_exp, kmeans_gene, kmeans_cell and kmeans_gene_cell
                  if <=1 then defaults to one group including all values
        k2 [int] - second number of groups for kmeans_gene_cell
                  if <=1 then defaults to one group including all values
        min_samples [int] - min number of samples; only for dbscan clustering
        n_pc and npc2 [None or int] - number of PCs to use before KMeans clustering
                           - NOTE: It is recommended to do this for methods: "kmeans_gene" and "kmeans_gene_cell"
        
    Returns
    -------
        groups [numpy array] - same dimension as G with values corresponding to group number (integer)
    '''
    # for auto k searches
    k_list = [2,3,4]
    
    if k == 1: # just one group
        groups = np.zeros(G.shape)
    else:
        if method == "cv_exp":
            cv_exp = np.nanstd(G, axis=0)/np.nanmean(G, axis=0)
            grouping = np.array_split(np.argsort(cv_exp), k) # split into ordered groups by index
            groups = np.ones(len(cv_exp))*np.nan # init nan group array
            for gi, g in enumerate(grouping): # assign group labels
                groups[g] = gi
            groups = np.tile(groups, (G.shape[0], 1)) # expand so each row is the same
        
        # grouping by genes only
        elif method == "kmeans_gene":
            X = StandardScaler().fit_transform(G.T)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            # if "auto", then select best k (k_gene)
            if k == 'auto':
                k = get_best_k(X, k_list)
            kmeans = KMeans(n_clusters=k, random_state=444).fit(X)
            groups = kmeans.labels_
            groups = np.tile(groups, (G.shape[0], 1)) # expand so each row is the same
        
        # grouping by cells only
        elif method == "kmeans_cell":
            X = StandardScaler().fit_transform(G)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            # if "auto", then select best k2 (k_cell)
            if k == 'auto':
                k = get_best_k(X, k_list)
            kmeans = KMeans(n_clusters=k, random_state=444).fit(X)
            groups = kmeans.labels_
            groups = np.tile(groups, (G.shape[1], 1)).T # expand so each col is the same
            
        # grouping by genes then by cells
        elif method == "kmeans_gene_cell":
            # gene grouping
            X = StandardScaler().fit_transform(G.T)
            if n_pc is not None:
                X = PCA(n_components=n_pc).fit_transform(X)
            # if "auto", then select best k (k_gene)
            if k == 'auto':
                k = get_best_k(X, k_list)
            kmeans_genes = KMeans(n_clusters=k, random_state=444).fit(X)
            cluster_genes = kmeans_genes.labels_
            groups = np.ones(G.shape)*np.nan # init groups array
            counter = 0 # to index new groups with integers
            # if "auto", then select best k2 (k_cell)
            if k2 == 'auto':
                X = StandardScaler().fit_transform(G)
                if n_pc is not None:
                    X = PCA(n_components=n_pc2).fit_transform(X)
                k2 = get_best_k(X, k_list)
            # within each gene group, group cells        
            for cg in np.unique(cluster_genes):
                G_group = G[:, cluster_genes==cg]
                X_group = StandardScaler().fit_transform(G_group)
                if n_pc2 is not None:
                    X_group = PCA(n_components=n_pc2).fit_transform(X_group)
                kmeans_cells = KMeans(n_clusters=k2, random_state=444).fit(X_group)
                cluster_cells = kmeans_cells.labels_
                for cc in np.unique(cluster_cells):
                    groups[np.ix_(cluster_cells==cc, cluster_genes==cg)] = counter
                    counter += 1
            
        else:
            raise Exception("method for get_grouping() is not recognized")
    
    return(groups, k, k2)


def get_best_k (X, k_list):
    '''
    Given a matrix X to perform KMeans clustering and list of k parameter values,
    searches for the best k value
    
    k_list should be in ascending order since get_best_k will terminate once the
    silhouette score decreases
    
    Parameters
    ----------
        X - array to perform K-means clustering on
        k_list - list of positive integers for number of clusters to use
        
    Returns
    -------
        best_k [int] - k value that returns the highest silhouette score
    '''
    from sklearn.metrics import silhouette_score
    
    # init search
    current_best = -np.inf
    best_k = 1
    
    # search along k_list
    for k in k_list:
        kmeans = KMeans(n_clusters=k, random_state=444).fit(X)
        score = silhouette_score(X, kmeans.labels_)
        if score > current_best: # update if score increases
            current_best = score
            best_k = k
        else: # stop if score decreases
            break
            
    return(best_k)



def conformalize_prediction_interval (adata, predicted, calib_genes, alpha_level=0.33, symmetric=True, return_scores_dict=False, compute_wasserstein=False):
    '''
    Builds conformal prediction interval sets for the predicted gene expression
    
    Parameters
    ----------
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm that corresponds to predicted gene expression 
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        alpha_level [float] - between 0 and 1; determines the alpha level; the CI will span the (1-alpha_level) interval
                              default value is alpha_level = 0.33 corresponding to 67% CI
        symmetric [bool] - whether to report symmetric prediction intervals or non-symmetric intervals; default is True (symmetric)
        return_scores_dict [bool] - whether to return the scores dictionary
        compute_wasserstein [bool] - whether to compute the Wasserstein distance of the score distributions between each subgroup and its calibration set
                                   - added to adata.obsm["{predicted}_wasserstein"]
                                   
    Returns
    -------
        Modifies adata in-place
        Optionally returns the scores_flattened_dict (dictionary containing calibration scores and group assignments)
    '''
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric)
    
    ### Building prediction intervals

    prediction_sets = (np.zeros(G.shape), np.zeros(G.shape)) # init prediction sets
    
    if compute_wasserstein is True: # set up matrix to store Wasserstein distances
        from scipy.stats import wasserstein_distance
        score_dist_wasserstein = np.ones(G.shape).astype(G.dtype)*np.nan

    # conformalize independently within groups of genes
    for group in np.unique(groups[~np.isnan(groups)]):
        
        # for symmetric intervals
        if symmetric is True:
            scores_flattened = scores_flattened_dict[str(group)] # flatten scores
            n = len(scores_flattened)
            if (n < 100): # if less than 100 samples in either set, then use the full group set
                scores_flattened = scores_flattened_dict[str(np.nan)]
                n = len(scores_flattened)-np.isnan(scores_flattened).sum()
            try:
                qhat = np.nanquantile(scores_flattened, np.ceil((n+1)*(1-alpha_level))/n)
            except:
                qhat = np.nan
            prediction_sets[0][groups==group] = (G-G_stdev*qhat)[groups==group] # lower bound
            prediction_sets[1][groups==group] = (G+G_stdev*qhat)[groups==group] # upper bound
        
        # for asymmetric intervals (Default)
        else:
            scores_lo_flattened = scores_flattened_dict[str(group)][0]
            scores_hi_flattened = scores_flattened_dict[str(group)][1]
            n_lo = len(scores_lo_flattened)-np.isnan(scores_lo_flattened).sum()
            n_hi = len(scores_hi_flattened)-np.isnan(scores_hi_flattened).sum()
            # compute qhat for lower and upper bounds
            if (n_lo < 100) or (n_hi < 100): # if less than 100 samples in either set, then use the full group set
                scores_lo_flattened = scores_flattened_dict[str(np.nan)][0]
                scores_hi_flattened = scores_flattened_dict[str(np.nan)][1]
                n_lo = len(scores_lo_flattened)-np.isnan(scores_lo_flattened).sum()
                n_hi = len(scores_hi_flattened)-np.isnan(scores_hi_flattened).sum()
            try:
                qhat_lo = np.nanquantile(scores_lo_flattened, np.ceil((n_lo+1)*(1-alpha_level))/n_lo)
                qhat_hi = np.nanquantile(scores_hi_flattened, np.ceil((n_hi+1)*(1-alpha_level))/n_hi)
            except:
                qhat_lo = np.nan
                qhat_hi = np.nan
            # compute bounds of prediction interval
            prediction_sets[0][groups==group] = (G-G_stdev*qhat_lo)[groups==group] # lower bound
            prediction_sets[1][groups==group] = (G+G_stdev*qhat_hi)[groups==group] # upper bound
            
        # Wasserstein distances
        if compute_wasserstein is True:
            # set up mask for calibration genes
            calib_idxs = [np.where(adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]
            calib_mask = np.full(G_stdev.shape, False)
            calib_mask[:,calib_idxs] = True
            # get CCV measures
            v = G_stdev[(groups==group)&~(calib_mask)].flatten() # group CCV
            if len(v) > 0: # skip if no observations in group
                if symmetric is True:
                    if n < 100:
                        u = G_stdev[calib_mask].flatten() # calibration CCV
                    else:
                        u = G_stdev[(groups==group)&(calib_mask)].flatten() # calibration CCV
                else:
                    if (n_lo < 100) or (n_hi < 100):
                        u = G_stdev[calib_mask].flatten() # calibration CCV
                    else:
                        u = G_stdev[(groups==group)&(calib_mask)].flatten() # calibration CCV
                # calculate wasserstein distance for the CCV distributions
                score_dist_wasserstein[groups==group] = wasserstein_distance(u, v).astype(G.dtype)
            
    # add prediction intervals to adata
    adata.uns['alpha'] = alpha_level
    adata.obsm[predicted+"_lo"] = pd.DataFrame(prediction_sets[0],
                                               columns=adata.obsm[predicted].columns,
                                               index=adata.obsm[predicted].index)
    adata.obsm[predicted+"_hi"] = pd.DataFrame(prediction_sets[1],
                                               columns=adata.obsm[predicted].columns,
                                               index=adata.obsm[predicted].index)
    # add wasserstein distances to adata        
    if compute_wasserstein is True:
        adata.obsm[predicted+"_wasserstein"] = pd.DataFrame(score_dist_wasserstein,
                                               columns=adata.obsm[predicted].columns,
                                               index=adata.obsm[predicted].index)
    
    
    if return_scores_dict is True:
    
        return(scores_flattened_dict)
        
        
        
def build_calibration_scores (adata, predicted, calib_genes, symmetric=False, include_zero_scores=False,
                              trim_quantiles=[None,None]):
    '''
    Builds calibration score sets
    
    Parameters
    ----------
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm with predicted gene expression values
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        include_zero_scores [bool] - whether to exclude zero scores
        trim_quantiles [list of len 2; None or float between 0 and 1] - specifies what quantile range of scores to trim to; None implies no bounds
        
    Returns
    -------
        scores_flattened_dict - dictionary containing the calibration scores for each stratified group
    '''
    
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)

    scores_flattened_dict = {}
    
    # get calibration genes
    calib_idxs = [np.where(adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]
    
    # iterate groups and build conformal sets of calibration scores
    for group in np.unique(groups[~np.isnan(groups)]):
        if (np.isnan(group)) or (group not in groups[:, calib_idxs]): # defer to using full calibration set
            scores_group = scores.copy()
            residuals_group = residuals.copy()
        else: # for groups that are found in the calibration set, build group-specific sets
            scores_group = scores.copy()[groups[:, calib_idxs]==group]
            residuals_group = residuals.copy()[groups[:, calib_idxs]==group]
        if symmetric is True: # symmetric calibration set
            if include_zero_scores is False:
                scores_flattened = scores_group[residuals_group != 0].flatten() # exclude zeros
            else:
                scores_flattened = scores_group.flatten()
            scores_flattened_dict[str(group)] = scores_flattened[np.isfinite(scores_flattened)] # add to dict
        else: # separate into hi/lo non-symmetric calibration sets
            if include_zero_scores is False:
                scores_lo_flattened = scores_group[residuals_group < 0].flatten()
                scores_hi_flattened = scores_group[residuals_group > 0].flatten()
            else:
                scores_lo_flattened = scores_group[residuals_group <= 0].flatten()
                scores_hi_flattened = scores_group[residuals_group >= 0].flatten()
            scores_flattened_dict[str(group)] = (scores_lo_flattened[np.isfinite(scores_lo_flattened)],
                                                 scores_hi_flattened[np.isfinite(scores_hi_flattened)]) # add to dict

    # build nan group consisting of all scores
    if symmetric is True: # symmetric calibration set
        if include_zero_scores is False:
            scores_flattened = scores[residuals != 0].flatten() # exclude zeros
        else:
            scores_flattened = scores.flatten()
        scores_flattened_dict[str(np.nan)] = scores_flattened[np.isfinite(scores_flattened)] # add to dict
    else: # separate into hi/lo non-symmetric calibration sets
        if include_zero_scores is False:
            scores_lo_flattened = scores[residuals < 0].flatten()
            scores_hi_flattened = scores[residuals > 0].flatten()
        else:
            scores_lo_flattened = scores[residuals <= 0].flatten()
            scores_hi_flattened = scores[residuals >= 0].flatten()
        scores_flattened_dict[str(np.nan)] = (scores_lo_flattened[np.isfinite(scores_lo_flattened)],
                                             scores_hi_flattened[np.isfinite(scores_hi_flattened)]) # add to dict
    
    # trim all scores if specified
    for key in scores_flattened_dict.keys():
    
        # determine quantiles from original scores
        if symmetric is True:
            if trim_quantiles[0] is not None:
                lower_bound = np.nanquantile(scores_flattened_dict[key], trim_quantiles[0])
            if trim_quantiles[1] is not None:
                upper_bound = np.nanquantile(scores_flattened_dict[key], trim_quantiles[1])
        else:
            if trim_quantiles[0] is not None:
                lower_bound_lo = np.nanquantile(scores_flattened_dict[key][0], trim_quantiles[0])
                lower_bound_hi = np.nanquantile(scores_flattened_dict[key][1], trim_quantiles[0])
            if trim_quantiles[1] is not None:
                upper_bound_lo = np.nanquantile(scores_flattened_dict[key][0], trim_quantiles[1])
                upper_bound_hi = np.nanquantile(scores_flattened_dict[key][1], trim_quantiles[1])
        
        # trim based on quantiles
        if symmetric is True:
            if trim_quantiles[0] is not None:    
                scores_flattened_dict[key] = scores_flattened_dict[key][scores_flattened_dict[key]>lower_bound]
            if trim_quantiles[1] is not None:    
                scores_flattened_dict[key] = scores_flattened_dict[key][scores_flattened_dict[key]<upper_bound]
        else:
            if trim_quantiles[0] is not None:    
                scores_flattened_dict[key] = (scores_flattened_dict[key][0][scores_flattened_dict[key][0]>lower_bound_lo],
                                              scores_flattened_dict[key][1][scores_flattened_dict[key][1]>lower_bound_hi])
            if trim_quantiles[1] is not None:    
                scores_flattened_dict[key] = (scores_flattened_dict[key][0][scores_flattened_dict[key][0]<upper_bound_lo],
                                              scores_flattened_dict[key][1][scores_flattened_dict[key][1]<upper_bound_hi])
       
    return (scores_flattened_dict)