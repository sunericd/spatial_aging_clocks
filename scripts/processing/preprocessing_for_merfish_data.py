'''
ARCHIVED SCRIPT
Runs processing of individual MERFISH samples, maps metadata, and combines into datasets

Inputs required: AnnData objects from scg_create_anndata.py

# - Converts the CSV count matrix files to AnnData format
# - Performs quality filtering of cells
# - Maps on age, mouse, and sample metadata values
# - Integrates data into AnnData objects

Conda environment used: `requirements/merfish.txt`
'''

import scanpy as sc
import squidpy as sq
import scrublet as scr
import pandas as pd
import numpy as np
import anndata as ad
import os

os.chdir("../..")



raw_dir = "data/anndata_raw"
save_dir = "data/anndata_processed"
min_volume = 100
min_count = 20
min_genes = 5
doublet_score_cutoff = 0.18


for fn in os.listdir(raw_dir):
    
    adata = sc.read_h5ad(os.path.join(raw_dir,fn))
    
    # remove "blank" genes
    non_blank_genes = [gene for gene in adata.var_names if "blank" not in gene.lower()]
    adata = adata[:, non_blank_genes].copy()
    
    # remove doublets using Scrublet
    scrub = scr.Scrublet(adata.X)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(log_transform=True)
    adata.obs['doublet_score'] = doublet_scores
    print(f"Dropped {np.sum(adata.obs['doublet_score']>doublet_score_cutoff)} doublets.")
    adata = adata[adata.obs['doublet_score']<doublet_score_cutoff,:].copy()
    
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
    '202301301605_MsBrain-VS62-OC4-OE3-BW_VMSC07101_region_0.h5ad': "OE3",
     '202301301605_MsBrain-VS62-OC4-OE3-BW_VMSC07101_region_1.h5ad': "OC4",
     '202302061111_MsBrain-VS62-YC4-OE4-BW_VMSC07101_region_0.h5ad': "OE4",
     '202302061111_MsBrain-VS62-YC4-OE4-BW_VMSC07101_region_1.h5ad': "YC4",
     '202302061153_MsBrain-VS62-19-30-BW_Beta10_region_0.h5ad': "30",
     '202302061153_MsBrain-VS62-19-30-BW_Beta10_region_1.h5ad': "19",
     '202302061221_MsBrain-VS62-11-38-BW_Beta8_region_0.h5ad': "38",
     '202302061221_MsBrain-VS62-11-38-BW_Beta8_region_1.h5ad': "11",
     '202302071943_MsBrain-VS62-YC3-OC3-BW_VMSC00201_region_0.h5ad': "OC3",
     '202302071943_MsBrain-VS62-YC3-OC3-BW_VMSC00201_region_1.h5ad': "YC3",
     '202302071954_MsBrain-VS62-1-46-BW_VMSC03501_region_0.h5ad': "46",
     '202302071954_MsBrain-VS62-1-46-BW_VMSC03501_region_1.h5ad': "1",
     '202302071956_MsBrain-VS62-7-42-BW_VMSC00701_region_0.h5ad': "42",
     '202302071956_MsBrain-VS62-7-42-BW_VMSC00701_region_1.h5ad': "7",
     '202302072052_MsBrain-VS62-2-BW_Beta10_region_0.h5ad': "2",
     '202302101157_MsBrain-VS62-14-33-BW_Beta10_region_0.h5ad': "14",
     '202302101157_MsBrain-VS62-14-33-BW_Beta10_region_1.h5ad': "33",
     '202302101312_MsBrain-VS62-39-BW_Beta8_region_0.h5ad': "39", # End of Batch 1
     '202308111122_MsBrain-62-VS85-YS_VMSC12502_region_0.h5ad': '62',
     '202308141252_MsBrain-VS85-86-70_VMSC07201_region_0.h5ad': '70',
     '202308141252_MsBrain-VS85-86-70_VMSC07201_region_1.h5ad': '86',
     '202308141339_MsBrain-VS85-53-101_Beta8_region_0.h5ad': '53', # split
     '202308141339_MsBrain-VS85-53-101_Beta8_region_1.h5ad': '101', # split
     '202308141340_MsBrain-VS85-80-75_VMSC12502_region_0.h5ad': '75',
     '202308141340_MsBrain-VS85-80-75_VMSC12502_region_1.h5ad': '80',
     '202308141358_MsBrain-VS85-61-93_VMSC16102_region_0.h5ad': '61', # split
     '202308141358_MsBrain-VS85-61-93_VMSC16102_region_1.h5ad': '93', # split
     '202308181352_MsBrain-VS85-Top-Young-Ctrl2_VMSC07101_region_0.h5ad': 'YC2',
     '202308181352_MsBrain-VS85-Top-Young-Ctrl2_VMSC07101_region_1.h5ad': 'OE2',
     '202308181451_MsBrain-VS85-Top-Young-Ctrl1_VMSC17502_region_0.h5ad': 'OC1',
     '202308181451_MsBrain-VS85-Top-Young-Ctrl1_VMSC17502_region_1.h5ad': 'YC1',
     '202308251220_MsBrain-VS85-Top-Old-Ctrl2_VMSC10802_region_0.h5ad': 'OC2',
     '202308251220_MsBrain-VS85-Top-Old-Ctrl2_VMSC10802_region_1.h5ad': 'OE1',
     '202308291027_MsBrain-VS85-Top-OT902_VMSC12502_region_0.h5ad': 'OT902',
     '202308291027_MsBrain-VS85-Top-OT902_VMSC12502_region_1.h5ad': 'OC903',
     '202309051110_MsBrain-VS85-TopOT1125_VMSC07101_region_0.h5ad': 'OT1125', # split
     '202309051110_MsBrain-VS85-TopOT1125_VMSC07101_region_1.h5ad': 'OC1138', # split
     '202309051447_MsBrain-VS85-Top-YC1989_VMSC17702_region_0.h5ad': 'YC1989',
     '202309051447_MsBrain-VS85-Top-YC1989_VMSC17702_region_1.h5ad': 'YC1975',
     '202309051527_MsBrain-VS85-Top-YC1990_VMSC13402_region_0.h5ad': 'YC1982',
     '202309051527_MsBrain-VS85-Top-YC1990_VMSC13402_region_1.h5ad': 'YC1990',
     '202309091203_MsBrain-VS85-Top-OT1084_VMSC13402_region_0.h5ad': 'OT1084', # split
     '202309091203_MsBrain-VS85-Top-OT1084_VMSC13402_region_1.h5ad': 'OC1083', # split
     '202309091203_MsBrain-VS85-Top-OT1160_VMSC16102_region_0.h5ad': 'OT1160', # split
     '202309091203_MsBrain-VS85-Top-OT1160_VMSC16102_region_1.h5ad': 'OC1226', # split
     '202309091208_MsBrain-VS85-68_VMSC14402_region_0.h5ad': '68',
     '202309141424_MsBrain-VS85-81-S2_VMSC07101_region_0.h5ad': '81',
     '202309141440_MsBrain-VS85-T57-B97_VMSC10802_region_0.h5ad': '57', # split
     '202309141440_MsBrain-VS85-T57-B97_VMSC10802_region_1.h5ad': '97', # split
     '202309150727_MsBrain-VS85-Top-89-S1_VMSC12502_region_0.h5ad': '89', # split
     '202309150727_MsBrain-VS85-Top-89-S1_VMSC12502_region_1.h5ad': '67', # split
     '202309220825_MsBrain-VS85-34_VMSC12502_region_0.h5ad': '34',
}

mouseid_to_slide_dict = {
     "OE3": 'A1',
     "OC4": 'A1',
     "OE4": 'A2',
     "YC4": 'A2',
     "30": 'A6',
     "19": 'A6',
     "38": 'A7',
     "11": 'A7',
     "OC3": 'A3',
     "YC3": 'A3',
     "46": 'A8',
     "1": 'A8',
     "7": 'A9',
     "42": 'A9',
     "2": 'A4',
     "14": 'A10',
     "33": 'A10',
     "39": 'A5', # End of Batch 1
     "62": 'B1',
     "70": 'B2',
     "86": 'B2',
     "53": 'B3',
     "101": 'B3',
     "75": 'B4',
     "80": 'B4',
     "61": 'B5',
     "93": 'B5',
     "YC2": 'B6',
     "OE2": 'B6',
     "OC1": 'B7',
     "YC1": 'B7',
     "OC2": 'B8',
     "OE1": 'B8',
     "OT902": 'B9',
     "OC903": 'B9',
     "OT1125": 'B10',
     "OC1138": 'B10',
     "YC1989": 'B11',
     "YC1975": 'B11',
     "YC1990": 'B12',
     "YC1982": 'B12',
     "OT1084": 'B13',
     "OC1083": 'B13',
     "OT1160": 'B14',
     "OC1226": 'B14',
     "68": 'B15',
     "81": 'B16',
     "57": 'B17',
     "97": 'B17',
     "89": 'B18',
     "67": 'B18',
     "34": 'B19',
}

mouseid_to_cohort_dict = {
     "OE3": "old_exercise",
     "OC4": "old_control",
     "OE4": "old_exercise",
     "YC4": "young_control",
     "30": "aging_coronal",
     "19": "aging_coronal",
     "38": "aging_coronal",
     "11": "aging_coronal",
     "OC3": "old_control",
     "YC3": "young_control",
     "46": "aging_coronal",
     "1": "aging_coronal",
     "7": "aging_coronal",
     "42": "aging_coronal",
     "2": "aging_sagittal",
     "14": "aging_coronal",
     "33": "aging_coronal",
     "39": "aging_sagittal", # End of Batch 1
     "62": "aging_sagittal",
     "70": "aging_coronal",
     "86": "aging_coronal",
     "53": "aging_coronal",
     "101": "aging_coronal",
     "75": "aging_coronal",
     "80": "aging_coronal",
     "61": "aging_coronal",
     "93": "aging_coronal",
     "YC2": "young_control",
     "OE2": "old_exercise",
     "OC1": "old_control",
     "YC1": "young_control",
     "OC2": "old_control",
     "OE1": "old_exercise",
     "OT902": "old_treatment",
     "OC903": "old_notreatment",
     "OT1125": "old_treatment",
     "OC1138": "old_notreatment",
     "YC1989": "young_notreatment",
     "YC1975": "young_notreatment",
     "YC1990": "young_notreatment",
     "YC1982": "young_notreatment",
     "OT1084": "old_treatment",
     "OC1083": "old_notreatment",
     "OT1160": "old_treatment",
     "OC1226": "old_notreatment",
     "68": "aging_sagittal",
     "81": "aging_sagittal",
     "57": "aging_coronal",
     "97": "aging_coronal",
     "89": "aging_coronal",
     "67": "aging_coronal",
     "34": "aging_sagittal",
}

mouseid_to_age_dict = {
     "OE3": 19,
     "OC4": 19,
     "OE4": 19,
     "YC4": 3,
     "30": 21.4,
     "19": 15.5,
     "38": 26.7,
     "11": 9.8,
     "OC3": 19,
     "YC3": 3,
     "46": 33.2,
     "1": 3.8,
     "7": 5.4,
     "42": 30.9,
     "2": 3.8,
     "14": 12.9,
     "33": 23.5,
     "39": 26.7, # End of Batch 1
     "62": 6.6,
     "70": 15.8,
     "86": 24.6,
     "53": 3.4,
     "101": 34.5,
     "75": 18.8,
     "80": 19.8,
     "61": 6.6,
     "93": 28.5,
     "YC2": 3,
     "OE2": 19,
     "OC1": 19,
     "YC1": 3,
     "OC2": 19,
     "OE1": 19,
     "OT902": 29.2,
     "OC903": 29.2,
     "OT1125": 27.3,
     "OC1138": 27.0,
     "YC1989": 4.8,
     "YC1975": 4.9,
     "YC1990": 4.8,
     "YC1982": 4.9,
     "OT1084": 27.9,
     "OC1083": 27.9,
     "OT1160": 26.5,
     "OC1226": 25.6,
     "68": 8.6,
     "81": 19.8,
     "57": 4.3,
     "97": 32.6,
     "89": 25.8,
     "67": 8.6,
     "34": 23.5,
}


# add mapped metadata and re-save anndata objects
dirname = "data/anndata_processed"

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

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    integrated_adata.append(adata)
    mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated.h5ad")


# Aging Coronal

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    if adata.obs['cohort'][0] == "aging_coronal":
        integrated_adata.append(adata)
        mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated_aging_coronal.h5ad")


# Aging Sagittal

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    if adata.obs['cohort'][0] == "aging_sagittal":
        integrated_adata.append(adata)
        mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated_aging_sagittal.h5ad")

# Exercise

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    if (adata.obs['cohort'][0] == "old_exercise") or (adata.obs['cohort'][0] == "old_control") or (adata.obs['cohort'][0] == "young_control"):
        integrated_adata.append(adata)
        mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated_exercise_coronal.h5ad")


# Reprogramming

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    if (adata.obs['cohort'][0] == "old_treatment") or (adata.obs['cohort'][0] == "old_notreatment") or (adata.obs['cohort'][0] == "young_notreatment"):
        integrated_adata.append(adata)
        mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated_reprogramming_coronal.h5ad")



# Aging Coronal + Exercise Coronal + Reprogramming Coronal

dirname = "data/anndata_processed"

integrated_adata = []
mouse_keys = []

for fn in os.listdir(dirname):
    adata = sc.read_h5ad(os.path.join(dirname,fn))
    if (adata.obs['cohort'][0] != "aging_sagittal"):
        integrated_adata.append(adata)
        mouse_keys.append(adata.obs["mouse_id"].values[0])

integrated_adata = ad.concat(integrated_adata, keys=mouse_keys, index_unique='-')
integrated_adata.write_h5ad("data/integrated_coronal.h5ad")




