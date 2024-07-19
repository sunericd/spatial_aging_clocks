# Spatiotemporal transcriptomic profiling and modeling of mouse brain at single-cell resolution reveals cell proximity effects of aging and rejuvenation
This repository contains all Jupyter notebooks and Python scripts for generating data and figures associated with the manuscript.

If you are interested in applying our spatial aging clocks to make predictions or to perform cell proximity effect analysis, check out our Python package where you can do this in a few lines of code: https://github.com/sunericd/SpatialAgingClock

## Quick start guide

### Data download and setup

First, clone our Github repository:

```
git clone https://github.com/sunericd/spatial_aging_clocks.git

cd spatial_aging_clocks
```

Then, download our publicly released datasets from (Zenodo link embargoed) and put the .h5ad files into the home directory of this repository.

Finally, to get the public release of our data formatted for use with these notebooks, just run:

```
python quick_setup.py
```

This will also unzip the directories that are needed to save plots, results, data, etc. Note that this will require an installation of the `anndata` package (any version will do) or you can set up one our conda environments (which will be needed to run the notebooks) following the instructions in the following section.

### Conda environments

To set up a conda environment for these analyses, we recommend installing all dependencies in a new conda environment and then setting that new environment as a jupyter kernel for use in the notebooks. Refer to ```requirements/merfish.txt``` for the main dependencies that we used in these notebooks ans scripts. For the GNN modeling, we use ```requirements/merfish_gnn.txt```. For TISSUE imputation, we use ```requirements/geneimputation.txt```. For some of the scripts running TISSUE imputation, we used a different conda environment following the TISSUE imputation package (please see https://github.com/sunericd/tissue-figures-and-analyses).


## Overview of notebooks

Jupyter notebooks containing code for making figures and running simulations for all results can be found in the top-level directory.

### 1. Cell type and region annotation (these are archival):
- ```1A_clustering_and_cell_types_aging.ipynb``` - cell type annotation and visualization for aging coronal dataset
- ```1A_clustering_and_cell_types_exercise.ipynb``` - cell type annotation and visualization for exercise coronal dataset
- ```1A_clustering_and_cell_types_reprogramming.ipynb``` - cell type annotation and visualization for partial reprogramming coronal dataset
- ```1A_clustering_and_cell_types_sagittal.ipynb``` - cell type annotation and visualization for sagittal coronal dataset
- ```1B_rotational_alignment.ipynb``` - manual rotational alignment of sections for aesthetic purposes
- ```1C_clustering_and_regions.ipynb``` - region and subregion annotation for coronal section datasets
- ```1D_metadata_integration.ipynb``` - generates supplementary table with sample information for all datasets


### 2. Cell type composition and gene expression changes:
- ```2A_cortical_layer_markers.ipynb``` - cortical layer marker analysis for cortical subregions
- ```2B_cell_type_composition_analysis.ipynb``` - cell type composition and proportion changes with age and with exercise/partial reprogramming
- ```2C_differential_gene_expression_analysis.ipynb``` - increasing/decreasing genes with age and visualization of GO biological process enrichment; differential gene expression analysis for exercise/reprogramming
- ```2D_gene_trajectory_analysis.ipynb``` - cell type-specific and region-specific gene expression trajectory fingerprints with age 


### 3. Spatial aging clocks
- ```3A_clocks_single_cell.ipynb``` - visualization and evaluation of spatial single-cell aging clocks
- ```3B_clocks_single_cell_regional.ipynb``` - visualization of region-specific spatial single-cell aging clocks and benchmark performance against global spatial aging clock
- ```3C_clocks_genes.ipynb``` - tabulation of features used in each spatial aging clock and GO term enrichment
- ```3D_clocks_cross_celltype_application.ipynb``` - evaluation of spatial aging clocks applied across different cell types
- ```3E_clocks_external_validation.ipynb``` - application of spatial aging clocks to validation datasets


### 4. Rejuvenation and adverse interventions
- ```4A_application_to_interventions.ipynb``` - application and evaluation of spatial aging clocks across different rejuvenating and adverse interventions
- ```4B_LPS_opc_oligodendrocyte_inflammation.ipynb``` - inflammation signature for oligodendrocytes and OPCs in LPS dataset
- ```4C_mean_imputation_alternative.ipynb``` - results using mean imputation for spatial aging clocks on interventions datasets


### 5. Cell proximity effects
- ```5A_cell_proximity_differential_expression.ipynb``` - differential expression and labeling of T cell and NSC marker genes for Near vs Far comparisons (regular proximity effect and area-restricted proximity effect)
- ```5A_cell_proximity_equality_of_variances.ipynb``` - Levene's test for age acceleration between near and far cells
- ```5A_regional_proximity_distance_cutoffs.ipynb``` - determination of unit distance cutoffs for proximity analysis
- ```5B_cell_proximity_aging_analysis.ipynb``` - calculation and analysis of proximity effects
- ```5C_cell_proximity_variations.ipynb``` - modified versions of proximity effect analysis (random matching, transcript count matching, area-restricted)
- ```5D_activation_inflammation_control.ipynb``` - activated microglia and inflamed oligodendrocyte control for T cell proximity effect
- ```5D_spatial_gradation_proximity_effect.ipynb``` - spatial gradation of key proximity effects with two methods
- ```5E_external_dataset_proximity_effects.ipynb``` - cell proximity effects computed for other datasets and aggregated
- ```5E_rejuvenation_intervention_proximity_effects.ipynb``` - cell proximity effects for exercise and partial reprogramming datasets separated by experiment conditions
- ```5F_GNN_perturbation_modeling.ipynb``` - GNN modeling for in silico perturbations of T cells and NSCs in local cell graphs


### 6. Potential mediators of proximity effects
- ```6A_cellular_proximity_imputation_setup.ipynb``` - setup datasets for TISSUE imputation of whole-transcriptome spatial gene expression and testing
- ```6B_cellular_proximity_imputation_QC.ipynb``` - evaluation of imputation performance and TISSUE calibration quality
- ```6C_cellular_proximity_imputation_dgea.ipynb``` - unbiased differential gene expression analysis on imputed expression for mediating pathways of proximity effects
- ```6C_cellular_proximity_imputation_signatures.ipynb``` - targeted imputed gene signatures analysis for mediating pathways of proximity effects
- ```6D_LPS_Ifng_proximity_effects.ipynb``` - cell proximity effects for the LPS dataset
- ```6E_immunofluorescence_proximity_analysis.ipynb``` - analysis of immunofluorescence images for validating cell proximity effect mediators


### 7. Reproducibility tests
- ```7A_batch_separated_reproducibility.ipynb``` - miscellaneous analyses using the two independent cohorts of the coronal section dataset separately
- ```7B_segmentation_spillover_checks.ipynb``` - miscellaneous analyses using a filtered subset of 220 genes with lower transcript misallocation (spillover) rates

## Overview of helper functions

These Python files contain helper functions that are imported across multiple notebooks and scripts to run shared analyses:

- ```ageaccel_proximity.py``` - methods for proximity effect analysis and age acceleration calculation
- ```clock_preprocessing.py``` - methods for preprocessing data to build and apply spatial aging clocks
- ```clock_prediction.py``` - methods for applying trained spatial aging clocks to new data
- ```spatial_propagation.py``` - methods for spatial smoothing and building spatial neighborhood graphs


## Overview of scripts

Python scripts for processing data, training clocks, running imputation, and generating intermediate data files are included in ```scripts```. **To run these files, see the slurm_jobs directories for example arguments, and run the script in their current directory**:
- ```train_clocks``` - contains scripts and job parameters for running cross-validated predictions with spatial aging clocks, training spatial aging clocks on the full datasets, and training region-specific aging clocks
- ```gnn_model``` - contains scripts and job parameters for running the dataset generation and training of the GNN model for predicting neighborhood aging
- ```imputation``` - contains scripts and job parameters for running TISSUE imputation and hypothesis testing for gene expression (```get_external_multi_ttest.py```) and for gene signatures (```get_external_multi_ttest_signatures.py```)
- ```processing``` - contains scripts for data processing and pre-processing including for creating AnnData objects from MERFISH data, preprocessing MERFISH data, and UMAP visualization with Leiden clustering. NOTE: These files are mostly archival since they rely on the raw datasets as input.
- ```go_enrichment``` - contains the R functions for running GO term enrichment for all increasing/decreasing genes with age, gene trajectories, clock genes, and differentially expressed genes with exercise. It also contains ```R_sessionInfo.txt``` with all package versions used for the analysis.
- ```region_dysfunction.py``` - script for running the region dysfunction analysis with age