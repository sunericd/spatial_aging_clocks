'''
Functions for running SpatialSmooth (spatial smoothing of gene expression by propagation across graphs)
'''

import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
import warnings


def update (X, Xt, S, alpha):
    '''
    Update equation shared by reinforce() and smooth()
    '''
    Xt1 = (1-alpha)*X + alpha*(S@Xt)
    return(Xt1)
    
    
def propagate (X, S, alpha, max_iter=100, tol=1e-2, verbose=True):
    '''
    Iterate update() until convergence is reached. See reinforce() and smooth() for usage/argument details
        X is the numpy matrix of node values to propagate
        S is the adjacency matrix
        
    verbose = whether to print propagation iterations
    '''
    # independent updates
    Xt = X.copy()
    Xt1 = update(X, Xt, S, alpha)

    iter_num = 1
    while (iter_num < max_iter) and np.any(np.divide(np.abs(Xt1-Xt), np.abs(Xt), out=np.full(Xt.shape,0.0), where=Xt!=0) > tol):
        Xt = Xt1
        Xt1 = update(X, Xt, S, alpha)
        iter_num += 1
    
    if verbose is True:
        print("Propagation converged after "+str(iter_num)+" iterations")
    
    return(Xt1)


def build_spatial_graph (adata, method="delaunay_radius", spatial="spatial", radius=None, n_neighbors=20, set_diag=True):
    '''
    Builds a spatial graph from AnnData according to specifications:
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
    
    Performs all computations inplace. Uses SquidPy implementations for graphs.
    '''
    if adata.shape[0] <= n_neighbors:
        n_neighbors=adata.shape[0]-1
    
    # delaunay graph
    if method == "delaunay": # triangulation only
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
    
    # radius-based methods
    elif method == "radius": # radius only
        if radius is None: # compute 90th percentile of delaunay triangulation
            sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic")
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        sq.gr.spatial_neighbors(adata, radius=radius, coord_type="generic", set_diag=set_diag)
    elif method == "delaunay_radius":
        sq.gr.spatial_neighbors(adata, delaunay=True, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0 # for computability
    elif method == "fixed_radius":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
        if radius is None:
            if isinstance(adata.obsp["spatial_distances"],np.ndarray):
                dists = adata.obsp['spatial_distances'].flatten()[adata.obsp['spatial_distances'].flatten() > 0]
            else:
                dists = adata.obsp['spatial_distances'].toarray().flatten()[adata.obsp['spatial_distances'].toarray().flatten() > 0]
            radius = np.percentile(dists, 75) + 1.5*(np.percentile(dists, 75) - np.percentile(dists, 25))
        adata.obsp['spatial_connectivities'][adata.obsp['spatial_distances']>radius] = 0
        adata.obsp['spatial_distances'][adata.obsp['spatial_distances']>radius] = 0 # for computability
            
    # fixed neighborhood size methods
    elif method == "fixed":
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighbors, coord_type="generic", set_diag=set_diag)
            
    else:
        raise Exception ("method not recognized")
        
        
def calc_adjacency_weights (adata, method="cosine", beta=0.0):
    '''
    Creates a normalized adjacency matrix containing edges weights for spatial graph
        adata [AnnData] = spatial data, must include adata.obsp['spatial_connectivities'] and adata.obsp['spatial_distances']
        method [str] = "binary" (weight is binary - 1 if edge exists, 0 otherwise); "cluster" (one weight for same-cluster and different weight for diff-cluster neighbors); "cosine" (weight based on cosine similarity between neighbor gene expressions)
        beta [float] = only used when method is "cluster"; between 0 and 1; specifies the non-same-cluster edge weight relative to 1 (for same cluster edge weight)
    
    Adds adata.obsp["S"]:
        S [numpy matrix] = normalized weight adjacency matrix; nxn where n is number of cells in adata
    '''
    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn.preprocessing import normalize
    
    # adjacency matrix from adata
    A = adata.obsp['spatial_connectivities']
    
    # compute weights
    if method == "binary":
        pass
    elif method == "cluster":
        # cluster AnnData if not already clustered
        if "cluster" not in adata.obs.columns:
            sc.tl.pca(adata)
            sc.pp.neighbors(adata, n_pcs=15)
            sc.tl.leiden(adata, key_added = "cluster")
        # init same and diff masks
        cluster_ids = adata.obs['cluster'].values
        same_mask = np.zeros(A.shape)
        for i in range(A.shape[1]):
            same_mask[:,i] = [1 if cid==cluster_ids[i] else 0 for cid in cluster_ids]
        diff_mask = np.abs(same_mask-1)
        # construct cluster-based adjacency matrix
        A = A*same_mask + A*diff_mask*beta
    elif method == "cosine":
        # PCA reduced space
        scaler = StandardScaler()
        pca = PCA(n_components=5, svd_solver='full')
        if isinstance(adata.X,np.ndarray):
            pcs = pca.fit_transform(scaler.fit_transform(adata.X))
        else:
            pcs = pca.fit_transform(scaler.fit_transform(adata.X.toarray()))
        # cosine similarities
        cos_sim = cosine_similarity(pcs)
        # update adjacency matrix
        A = A*cos_sim
        A[A < 0] = 0
    else:
        raise Exception ("weighting must be 'binary', 'cluster', 'cosine'")
    
    # normalized adjacency matrix
    S = normalize(A, norm='l1', axis=1)
    
    # update adata
    adata.obsp["S"] = S