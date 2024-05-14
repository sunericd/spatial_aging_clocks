'''
ARCHIVED SCRIPT
Runs preprocessing and initial Leiden clustering for combined datasets 

Inputs required: integrated AnnData objects from preprocessing_for_merfish_data.py

Conda environment used: `requirements/merfish.txt`
'''


import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import anndata as ad
import os
from scipy.stats import pearsonr, spearmanr

os.chdir("../..")


# DATASETS WITH MULTIPLE BATCHES

filenames_to_prepare = ["data/integrated_aging_coronal.h5ad", "data/integrated_exercise_coronal.h5ad", "data/integrated_aging_sagittal.h5ad"]

for fn in filenames_to_prepare:

    # Read in AnnData
    adata = sc.read_h5ad(fn)

    # Normalization of data by volume
    adata.X = adata.X/np.array(adata.obs['volume'])[:,None]

    # Remove top 2% and bottom 2% of total expression cells
    top = np.percentile(adata.X.sum(axis=1), 98)
    bottom = np.percentile(adata.X.sum(axis=1), 2)
    adata = adata[(adata.X.sum(axis=1) < top) & (adata.X.sum(axis=1) > bottom)].copy()

    adata.write_h5ad(fn.split(".")[0]+"_raw.h5ad")

    # Normalize total to 250
    sc.pp.normalize_total(adata, target_sum=250)

    # Log transform
    sc.pp.log1p(adata)

    # Z-score (need to do)
    sc.pp.scale(adata, max_value=10)


    # run pca
    sc.tl.pca(adata)

    # add batch variable (independent)
    adata.obs["batch"] = [x[0] for x in adata.obs["slide_id"]]

    # Neighbors and UMAP
    sc.external.pp.bbknn(adata) # use inplace of sc.pp.neighbors()
    sc.tl.umap(adata)

    adata.write_h5ad(fn.split(".")[0]+"_bbknn_umap.h5ad")

    # Clustering
    sc.tl.leiden(adata, key_added="clusters", resolution=0.5)
    adata.write_h5ad(fn.split(".")[0]+"_bbknn_umap_leiden1.h5ad")



# DATASETS WITHOUT INTEGRATION

nonintegrate_filenames = ["data/integrated_reprogramming_coronal.h5ad"]

for fn in nonintegrate_filenames:

    # Read in AnnData
    adata = sc.read_h5ad(fn)

    # Normalization of data by volume
    adata.X = adata.X/np.array(adata.obs['volume'])[:,None]

    # Remove top 2% and bottom 2% of total expression cells
    top = np.percentile(adata.X.sum(axis=1), 98)
    bottom = np.percentile(adata.X.sum(axis=1), 2)
    adata = adata[(adata.X.sum(axis=1) < top) & (adata.X.sum(axis=1) > bottom)].copy()

    adata.write_h5ad(fn.split(".")[0]+"_raw.h5ad")

    # Normalize total to 250
    sc.pp.normalize_total(adata, target_sum=250)

    # Log transform
    sc.pp.log1p(adata)

    # Z-score (need to do)
    sc.pp.scale(adata, max_value=10)


    # run pca and Harmony integration of aging and exercise
    sc.tl.pca(adata)

    # Neighbors and UMAP
    sc.pp.neighbors(adata, n_pcs=20, n_neighbors=15) # build graph from PCA that will be used for UMAP and Leiden clustering
    sc.tl.umap(adata)

    adata.write_h5ad(fn.split(".")[0]+"_bbknn_umap.h5ad")

    # Clustering
    sc.tl.leiden(adata, key_added="clusters", resolution=0.5)
    adata.write_h5ad(fn.split(".")[0]+"_umap_leiden1.h5ad")