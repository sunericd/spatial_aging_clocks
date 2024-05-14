'''
ARCHIVED SCRIPT:
Runs the re-generation of UMAP plots after removing the false cell cluster (bead segmentations)
The released datasets already have the cell cluster removed and the new UMAPs.

Inputs required: Intermediate outputs of the 1A_*.ipynb notebooks for cell type annotation

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

# CORONAL
adata = sc.read_h5ad("data/integrated_aging_coronal_clustered.h5ad")
adata = adata[(adata.obs.clusters!="1")&(adata.obs.mouse_id!="89")&(adata.obs.mouse_id!="67")].copy() # exclude bad cluster and slide
sc.tl.umap(adata)
adata.write_h5ad("data/integrated_aging_coronal_clustered.h5ad")

# update other data objects with new UMAP
fns = ["dataupload/data/integrated_aging_coronal_celltyped_regioned_raw.h5ad"]

for fn in fns:
    raw_adata = sc.read_h5ad(fn)
    raw_adata.uns['umap'] = adata.uns['umap'].copy()
    raw_adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    raw_adata.write_h5ad(fn)
    
 
# SAGITTAL 
adata = sc.read_h5ad("data/integrated_aging_sagittal_clustered.h5ad")
adata = adata[(adata.obs.clusters!="1")].copy() # exclude bad cluster and slide
sc.tl.umap(adata)
adata.write_h5ad("data/integrated_aging_sagittal_clustered.h5ad")

# update other data objects with new UMAP
fns = ["dataupload/data/integrated_aging_sagittal_clustered_registered_raw.h5ad"]

for fn in fns:
    raw_adata = sc.read_h5ad(fn)
    raw_adata.uns['umap'] = adata.uns['umap'].copy()
    raw_adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    raw_adata.write_h5ad(fn)



# EXERCISE
adata = sc.read_h5ad("data/integrated_exercise_coronal_clustered.h5ad")
adata = adata[(adata.obs.clusters!="0")].copy() # exclude bad cluster and slide
sc.tl.umap(adata)
adata.write_h5ad("data/integrated_exercise_coronal_clustered.h5ad")

# update other data objects with new UMAP
fns = ["dataupload/data/integrated_exercise_coronal_celltyped_regioned_raw.h5ad"]

for fn in fns:
    raw_adata = sc.read_h5ad(fn)
    raw_adata.uns['umap'] = adata.uns['umap'].copy()
    raw_adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    raw_adata.write_h5ad(fn)



# REPROGRAMMING
adata = sc.read_h5ad("data/integrated_reprogramming_coronal_clustered.h5ad")
adata = adata[(adata.obs.clusters!="0")].copy() # exclude bad cluster and slide
sc.tl.umap(adata)
adata.write_h5ad("data/integrated_reprogramming_coronal_clustered.h5ad")

# update other data objects with new UMAP
fns = ["dataupload/data/integrated_reprogramming_coronal_celltyped_regioned_raw.h5ad"]

for fn in fns:
    raw_adata = sc.read_h5ad(fn)
    raw_adata.uns['umap'] = adata.uns['umap'].copy()
    raw_adata.obsm['X_umap'] = adata.obsm['X_umap'].copy()
    raw_adata.write_h5ad(fn)