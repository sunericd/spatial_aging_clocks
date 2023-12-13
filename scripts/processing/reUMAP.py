import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import anndata as ad
import os
from scipy.stats import pearsonr, spearmanr

adata = sc.read_h5ad("data/integrated_aging_coronal_clustered.h5ad")
# adata = adata[(adata.obs.clusters!="1")&(adata.obs.mouse_id!="89")&(adata.obs.mouse_id!="67")].copy() # exclude bad cluster and slide
# sc.tl.umap(adata)
# adata.write_h5ad("data/integrated_aging_coronal_clustered.h5ad")

# update other data objects with new UMAP
fns = ["dataupload/integrated_aging_coronal_celltyped_regioned_raw.h5ad"]

for fn in fns:
    raw_adata = sc.read_h5ad(fn)
    #raw_adata = raw_adata[(raw_adata.obs.clusters!="1")&(raw_adata.obs.mouse_id!="89")&(raw_adata.obs.mouse_id!="67")].copy() # exclude bad cluster and slide
    raw_adata.uns['umap'] = adata.uns['umap'].copy()
    raw_adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    raw_adata.write_h5ad(fn)