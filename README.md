# Spatiotemporal transcriptomic profiling and modeling of mouse brain at single-cell resolution reveals cell proximity effects of aging and rejuvenation
This repository contains all Jupyter notebooks and Python scripts for generating data and figures associated with the manuscript.

To set up a conda environment for these analyses, we recommend installing all dependencies in a new conda environment and then setting that new environment as a jupyter kernel for use in the notebooks. Refer to ```environment.yml``` for the main dependencies that we used in these notebooks. For some of the scripts running TISSUE imputation, we used a different conda environment (please see https://github.com/sunericd/tissue-figures-and-analyses)

Jupyter notebooks containing code for making figures and running simulations for all results can be found in the top-level directory.
- ```A1_clustering_and_cell_types_aging.ipynb``` - cell type annotation and visualization for aging coronal dataset
- ```A1_clustering_and_cell_types_exercise.ipynb``` - cell type annotation and visualization for exercise coronal dataset
- ```A1_clustering_and_cell_types_reprogramming.ipynb``` - cell type annotation and visualization for partial reprogramming coronal dataset
- ```A1_clustering_and_cell_types_sagittal.ipynb``` - cell type annotation and visualization for sagittal coronal dataset
- ```A2_rotational_alignment.ipynb``` - manual rotational alignment of sections for aesthetic purposes
- ```A3_clustering_and_regions.ipynb``` - region and subregion annotation for coronal section datasets
- ```B1_cell_type_composition_analysis.ipynb``` - cell type composition and proportion changes with age and with exercise/partial reprogramming
- ```B2_differential_gene_expression_analysis.ipynb``` - differential gene expression with age
- ```B3_gene_trajectory_analysis.ipynb``` - cell type-specific and region-specific gene expression trajectory fingerprints with age 
- ```B4_metadata_integration.ipynb``` - generates supplementary table with sample information for all datasets
- ```C1_clocks_single_cell.ipynb``` - visualization and evaluation of spatial single-cell aging clocks
- ```C2_clocks_single_cell_regional.ipynb``` - visualization of region-specific spatial single-cell aging clocks and benchmark performance against global spatial aging clock
- ```C3_clocks_genes.ipynb``` - tabulation of features used in each spatial aging clock
- ```C4_clocks_cross_celltype_application.ipynb``` - evaluation of spatial aging clocks applied across different cell types
- ```C5_clocks_external_validation.ipynb``` - application of spatial aging clocks to validation datasets and interventions datasets
- ```D1_regional_proximity_distance_cutoffs.ipynb``` - determination of unit distance cutoffs for proximity analysis
- ```D2_cell_proximity_aging_analysis.ipynb``` - calculation and analysis of proximity effects
- ```D3_cellular_proximity_imputation_setup.ipynb``` - setup datasets for TISSUE imputation of whole-transcriptome spatial gene expression and testing
- ```D4_cellular_proximity_imputation_dgea.ipynb``` - unbiased differential gene expression analysis on imputed expression for mediating pathways of proximity effects
- ```D4_cellular_proximity_imputation_QC.ipynb``` - evaluation of imputation performance and TISSUE calibration quality
- ```D4_cellular_proximity_imputation_signatures.ipynb``` - targeted imputed gene signatures analysis for mediating pathways of proximity effects

Python scripts for processing data, training clocks, running imputation, and generating intermediate data files are included in ```scripts```:
- ```clocks``` - contains scripts and job parameters for running cross-validated predictions with spatial aging clocks, training spatial aging clocks on the full datasets, and training region-specific aging clocks
- ```imputation``` - contains scripts and job parameters for running TISSUE imputation and hypothesis testing for gene expression (```get_external_multi_ttest.py```) and for gene signatures (```get_external_multi_ttest_signatures.py```)
- ```processing``` - contains scripts for data processing and pre-processing including for creating AnnData objects from MERFISH data, preprocessing MERFISH data, UMAP + Leiden clustering, and processing and mapping of metadata to the 140-gene pilot MERFISH dataset
- ```ageaccel_proximity.py``` - gmethods for proximity effect analysis and age acceleration calculation
- ```clock_preprocessing.py``` - methods for preprocessing data to build and apply spatial aging clocks
- ```spatial_propagation.py``` - useful methods for spatial smoothing and building spatial neighborhood graphs
