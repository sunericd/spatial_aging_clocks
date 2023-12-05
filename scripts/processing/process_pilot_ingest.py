import scanpy as sc
import squidpy as sq
import pandas as pd
import numpy as np
import anndata as ad
import os
from scipy.stats import pearsonr, spearmanr
import scrublet as scr

os.chdir("/labs/abrunet1/Eric/MERFISH")


celltype_palette = {'Neuron-Excitatory':'forestgreen',
                    'Neuron-Inhibitory':'lightgreen', 
                    'Neuron-MSN':'yellowgreen',
                    'Astrocyte': 'royalblue', 
                    'Microglia': 'aqua', 
                    'Oligodendrocyte': 'skyblue', 
                    'OPC': 'deepskyblue',
                    'Endothelial': 'red', 
                    'Pericyte': 'darkred', 
                    'VSMC': 'salmon', 
                    'VLMC': 'indianred',
                    'Ependymal': 'gray', 
                    'Neuroblast': 'sandybrown', 
                    'NSC':'darkorange', 
                    'Macrophage':'purple', 
                    'Neutrophil':'darkviolet',
                    'T cell':'magenta', 
                    'B cell':'orchid',
}



fn = "data/pilot_merfish_2022/integrated.h5ad"

# Read in AnnData
adata = sc.read_h5ad(fn)

# Normalization of data by volume
adata.X = adata.X/np.array(adata.obs['volume'])[:,None]

# Remove top 2% and bottom 2% of total expression cells
top = np.percentile(adata.X.sum(axis=1), 98)
bottom = np.percentile(adata.X.sum(axis=1), 2)
adata = adata[(adata.X.sum(axis=1) < top) & (adata.X.sum(axis=1) > bottom)]

adata.write_h5ad(fn.split(".")[0]+"_raw.h5ad")

# Normalize total to 250
sc.pp.normalize_total(adata, target_sum=250)

# Log transform
sc.pp.log1p(adata)

# Z-score (need to do)
sc.pp.scale(adata, max_value=10)


# PCA
sc.tl.pca(adata)




# Scanpy Ingest for mapping on cell type annotations


# read in data
adata_ref = sc.read_h5ad("data/integrated_aging_coronal_celltyped_regioned_raw.h5ad")
adata_ref = adata_ref[(adata_ref.obs.clusters!="1")&(adata_ref.obs.mouse_id!="89")&(adata_ref.obs.mouse_id!="67")]


var_names = adata_ref.var_names.intersection(adata.var_names)
adata_ref = adata_ref[:, var_names]
adata = adata[:, var_names]


# PCA on ref
sc.tl.pca(adata_ref)
sc.pp.neighbors(adata_ref)
sc.tl.umap(adata_ref)


# to fix ingest bug: https://github.com/scverse/scanpy/issues/2085
b = np.array(list(map(len, adata_ref.obsp['distances'].tolil().rows))) # number of neighbors of each cell
adata_ref_subset = adata_ref[np.where(b == 15-1)[0]] # select only those with the right number
sc.pp.neighbors(adata_ref_subset, 15) # rebuild the neighbor graph


sc.tl.ingest(adata, adata_ref_subset, obs='celltype')


adata.write_h5ad("data/pilot_merfish_2022/integrated_ingested.h5ad")






