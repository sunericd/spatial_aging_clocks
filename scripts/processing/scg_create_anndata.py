'''
ARCHIVED SCRIPT
Runs processing of Vizgen lab service data outputs into AnnData format

Inputs required: "analyzed_data" directory with Vizgen lab service outputs

Conda environment used: `requirements/merfish.txt`
'''


# code to run on SCG
import anndata as ad
import scanpy as sc
import pandas as pd
import os
import numpy as np

os.chdir("../..")


# NOTE: Block comment/uncomment the sections to process different datasets



# For Batch 1
top_dir = "analyzed_data"
seg_dir = "Cellpose_DAPI_polyT"
data_fn = "cellpose_cell_by_gene.csv"
meta_fn = "cellpose_cell_metadata.csv"

# search through directories
for slide in os.listdir(top_dir):
    if os.path.isdir(os.path.join(top_dir,slide)):
        for region in os.listdir(os.path.join(top_dir,slide)):
            if ("region" in region) and (os.path.isdir(os.path.join(top_dir,slide,region))):
                pathdir = os.path.join(top_dir,slide,region,seg_dir)
                
                # read in and make anndata object
                adata = sc.read_csv(os.path.join(pathdir,data_fn), first_column_names=True)
                metadata = pd.read_csv(os.path.join(pathdir,meta_fn), index_col=0)
                metadata.index = adata.obs_names
                adata.obs = metadata
                adata.obsm['spatial'] = adata.obs[["center_x", "center_y"]].to_numpy()
                
                # save anndata
                save_name = f"{slide}_{region}.h5ad"
                adata.write_h5ad(os.path.join("anndata",save_name))
                
# For Batch 2
top_dir = "batch2_data"
seg_dir = "Cellpose"
data_fn = "cellpose_cell_by_gene.csv"
meta_fn = "cellpose_cell_metadata.csv"

# search through directories
for slide in os.listdir(top_dir):
    if os.path.isdir(os.path.join(top_dir,slide)):
        for region in os.listdir(os.path.join(top_dir,slide)):
            if ("region" in region) and (os.path.isdir(os.path.join(top_dir,slide,region))):
                pathdir = os.path.join(top_dir,slide,region,seg_dir)
                
                # read in and make anndata object
                adata = sc.read_csv(os.path.join(pathdir,data_fn), first_column_names=True)
                metadata = pd.read_csv(os.path.join(pathdir,meta_fn), index_col=0)
                metadata.index = adata.obs_names
                adata.obs = metadata
                adata.obsm['spatial'] = adata.obs[["center_x", "center_y"]].to_numpy()
                
                # save anndata
                save_name = f"{slide}_{region}.h5ad"
                adata.write_h5ad(os.path.join("anndata",save_name))
