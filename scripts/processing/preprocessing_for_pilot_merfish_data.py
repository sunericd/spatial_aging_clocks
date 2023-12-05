# # Preprocessing of Pilot (140-gene) MERFISH data for downstream analysis
# 
# - Converts the CSV count matrix files to AnnData format
# - Performs quality filtering of cells
# - Maps on age, mouse, and sample metadata values
# - Integrates data into AnnData objects


import scanpy as sc
import squidpy as sq
import scrublet as scr
import pandas as pd
import numpy as np
import anndata as ad
import os


raw_dir = "data/pilot_merfish_2022/anndata_raw"
save_dir = "data/pilot_merfish_2022/anndata_processed"
min_volume = 20
min_count = 20
min_genes = 5
doublet_score_cutoff = 0.18


for fn in os.listdir(raw_dir):
    
    adata = sc.read_h5ad(os.path.join(raw_dir,fn))
    
    # remove "blank" genes
    non_blank_genes = [gene for gene in adata.var_names if "blank" not in gene.lower()]
    adata = adata[:, non_blank_genes].copy()
    
    # compute number of detected genes and counts
    adata.obs['num_detected_genes'] = np.count_nonzero(adata.X, axis=1)
    adata.obs['barcodeCount'] = np.sum(adata.X, axis=1)
    
    max_volume = 3*np.median(adata.obs.volume)
    
    # filter cells based on volume and transcript count
    prev_len = adata.shape[0]
    adata = adata[(adata.obs.volume > min_volume) & 
                  (adata.obs.barcodeCount > min_count) & 
                  (adata.obs.volume < max_volume) &
                  (adata.obs.num_detected_genes > min_genes), :].copy()
    print(f"Filtered from {prev_len} to {adata.shape[0]} cells.")
    
    # save processed anndata
    adata.write_h5ad(os.path.join(save_dir,fn))


# ### Map on additional metadata to anndata objects

# dicts for mapping

file_to_mouseid_dict = {
    'MsBrain_VS38_Middle-4_QC_VA79GMblock_V1_BW_4-4-2022_region_0.h5ad': "Middle1",
     'MsBrain_VS38_Middle-4_QC_VA79GMblock_V1_BW_4-4-2022_region_1.h5ad': "Middle2",
     'MsBrain_VS38_Old-4_QC_VA79GMblock_V1_BW_4-5-2022_region_0.h5ad': "Old1",
     'MsBrain_VS38_Old-4_QC_VA79GMblock_V1_BW_4-5-2022_region_1.h5ad': "Old2",
     'MsBrain_VS38_Young-5_QC_VA79GMblock_V3_BW_4-5-2022_region_0.h5ad': "Young1",
     'MsBrain_VS38_Young-5_QC_VA79GMblock_V3_BW_4-5-2022_region_1.h5ad': "Young2",
}

mouseid_to_slide_dict = {
     "Middle1": 1,
     "Middle2": 1,
     "Old1": 2,
     "Old2": 2,
     "Young1": 3,
     "Young2": 3,
}

mouseid_to_cohort_dict = {
     "Middle1": "Middle",
     "Middle2": "Middle",
     "Old1": "Old",
     "Old2": "Old",
     "Young1": "Young",
     "Young2": "Young",
}

mouseid_to_age_dict = {
     "Middle1": 19,
     "Middle2": 19,
     "Old1": 25,
     "Old2": 25,
     "Young1": 3,
     "Young2": 3,
}


# add mapped metadata and re-save anndata objects
dirname = "data/pilot_merfish_2022/anndata_processed"

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    
    # map mouseid from filename and get metadata
    mouseid = file_to_mouseid_dict[fn]
    slide = mouseid_to_slide_dict[mouseid]
    cohort = mouseid_to_cohort_dict[mouseid]
    age = mouseid_to_age_dict[mouseid]
    
    # add to anndata.obs
    adata.obs["mouse_id"] = mouseid
    adata.obs["slide_id"] = slide
    adata.obs["cohort"] = cohort
    adata.obs["age"] = age
    
    # save processed anndata
    adata.write_h5ad(os.path.join(dirname,fn))


# ### Integrating samples into one AnnData object

# All

dirname = "data/pilot_merfish_2022/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    integrated_adata.append(adata)
    mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/pilot_merfish_2022/integrated.h5ad")




