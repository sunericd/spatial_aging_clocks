'''
Makes directory structure for notebooks
Reformats the public release of data to be compatible with notebooks
'''

import anndata as ad
import numpy as np
import os

# Make directory structure
directories_to_make = ["data/",
                       "data/allen_2022_aging_merfish",
                       "data/androvic_2023_injury_merfish",
                       "data/external_RNAseq",
                       "data/external_RNAseq/ximerakis_2019_scRNAseq",
                       "data/gnn_datasets",
                       "data/IHC_qupath_outputs",
                       "data/kukanja_EAE_MS_2024",
                       "data/pilot_merfish_2022",
                       "data/zeng_2023_alzheimer_starmap",
                       "plots/",
                       "plots/ageaccel",
                       "plots/cell_composition",
                       "plots/clocks",
                       "plots/clocks/pseudobulked",
                       "plots/clocks/single_cell",
                       "plots/clustering",
                       "plots/dgea",
                       "plots/exercise",
                       "plots/gene_trajectory",
                       "plots/gnn",
                       "plots/ihc",
                       "plots/mechanism",
                       "plots/pathways",
                       "plots/proximity",
                       "plots/QC",
                       "plots/reprogramming",
                       "results/",
                       "results/cell_composition",
                       "results/clocks",
                       "results/clocks/anndata",
                       "results/clocks/applied",
                       "results/clocks/mean_impute_files",
                       "results/clocks/stats",
                       "results/dgea",
                       "results/for_imputation",
                       "results/gene_trajectory",
                       "results/gnn",
                       "results/pathway_enrichment",
                       "results/preprocessing",
                       "results/proximity"]

print ("Making directory structure...")
for directory in directories_to_make:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Reformat datasets

public_filenames = ["aging_coronal",
                    "aging_sagittal",
                    "exercise",
                    "reprogramming"]

new_filenames = ["integrated_aging_coronal_celltyped_regioned_raw",
                 "integrated_aging_sagittal_clustered_registered_raw",
                 "integrated_exercise_coronal_celltyped_regioned_raw",
                 "integrated_reprogramming_coronal_celltyped_regioned_raw"]

print ("Reformatting datasets...")
for i in range(len(public_filenames)):

    # Read in public data (should have in data/ folder)
    adata = ad.read_h5ad(f"{public_filenames[i]}.h5ad")

    # Map region and subregion keys
    adata.obs["region"] = adata.obs.subregion
    del adata.obs["subregion"]

    # Normalization of data by volume
    adata.X = adata.X/np.array(adata.obs['volume'])[:,None]
    
    # Save new data file
    adata.write_h5ad(f"data/{new_filenames[i]}.h5ad")
    print(f"reformatted dataset saved at data/{new_filenames[i]}.h5ad")
    
print("DONE!")