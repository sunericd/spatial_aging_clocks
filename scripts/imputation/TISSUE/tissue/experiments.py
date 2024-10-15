# Contains compound functions for generating results for experiments with TISSUE
# These are unlikely to be used for general applications but were used in our development/testing of TISSUE

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
import squidpy as sq
import anndata as ad
import warnings
import os
import gc

#from tissue.utils import large_save, large_load
from .utils import large_save, large_load
from .main import load_spatial_data, conformalize_prediction_interval, get_spatial_uncertainty_scores_from_metadata
from .downstream import multiple_imputation_testing


def group_conformalize_from_intermediate(dataset_name, methods, symmetric, alpha_levels,
                                         save_alpha=[0.05], savedir="SCPI", type_dataset="DataUpload"):
    '''
    Function for taking intermediate fold predictions and running group conformalization for all different alpha values
    
    Returns a results dictionary with calibration quality (res_dict) and the AnnData with CI for all folds at alpha of save_alpha [float]
    
    Parameters
    ----------
        dataset_name [str] - name of folder in DataUpload/
        methods [list of str] - list of method keys to use for prediction_sets
        symmetric [bool] - whether to use symmetric prediction intervals
        alpha_levels [array] - alpha levels to calibrate over
        save_alpha [list of float] - alphas to save prediction intervals into adata.obsm
        savedir [str] - folder where the intermediate results are saved (independent folds)
        type_dataset [str] - default to "DataUpload" but may have additional options in the future
        
    Returns
    -------
        res_dict [dict] - dictionary of calibration statistics / coverage statistics across the alpha levels
        adata [AnnData] - anndata with calibration results added to metadata
    '''
    # read in spatial data
    if type_dataset == "DataUpload":
        if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
            adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                      "DataUpload/"+dataset_name+"/Locations.txt",
                                       spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
        else:
            adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                      "DataUpload/"+dataset_name+"/Locations.txt")
    else:
        adata = sc.read_h5ad(os.path.join("additional_data",dataset_name,"spatial.h5ad"))
    adata.var_names = [x.lower() for x in adata.var_names]
    
    # results dict
    res_dict = {}
    
    for method in methods:

        res_dict[method] = {}
        res_dict[method]['ind_gene_results'] = {}

        calibration_weight = 0 # for computing weighted average
        test_weight = 0

        dirpath = savedir+"/"+dataset_name+"_intermediate/"+method
        
        folds = np.load(os.path.join(savedir+"/"+dataset_name+"_intermediate/"+method,"folds.npy"), allow_pickle=True)

        # subset spatial data into shared genes
        gene_names = np.concatenate(folds)
        adata = adata[:, gene_names]

        for i, fold in enumerate(folds):

            # load adata within fold
            sub_adata = large_load(os.path.join(dirpath, "fold"+str(i)))
            target_genes = list(fold)

            # subset data
            predicted = method+"_predicted_expression"
            test_genes = target_genes.copy()
            calib_genes = [gene for gene in gene_names if gene not in test_genes]
            test_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in test_genes]
            calib_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]

            # get uncertainties and scores from saved adata
            scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (sub_adata, predicted)

            # init dict for individual gene results
            for g in test_genes:
                if g not in res_dict[method]['ind_gene_results'].keys():
                    res_dict[method]['ind_gene_results'][g] = {}
                    res_dict[method]['ind_gene_results'][g]['1-alpha'] = 1-alpha_levels
                    res_dict[method]['ind_gene_results'][g]['test'] = []

            # iterate over different alphas for conformalization
            test_perc = []
            calib_perc = []

            for alpha_level in alpha_levels:
                sub_adatac = sub_adata.copy()
                conformalize_prediction_interval (sub_adatac, predicted, calib_genes, alpha_level=alpha_level,
                                                  symmetric=symmetric, return_scores_dict=False)
                
                prediction_sets = (sub_adatac.obsm[predicted+"_lo"].values, sub_adatac.obsm[predicted+"_hi"].values)
                
                test_perc.append(np.nanmean(((adata[:,test_genes].X>prediction_sets[0][:,test_idxs]) & (adata[:,test_genes].X<prediction_sets[1][:,test_idxs]))[(G[:,test_idxs]!=0)&(G_stdev[:,test_idxs]!=0)&(adata[:,test_genes].X!=0)]))
                calib_perc.append(np.nanmean(((adata[:,calib_genes].X>prediction_sets[0][:,calib_idxs]) & (adata[:,calib_genes].X<prediction_sets[1][:,calib_idxs]))[(G[:,calib_idxs]!=0)&(G_stdev[:,calib_idxs]!=0)&(adata[:,calib_genes].X!=0)]))

                # Compute individual calibration curves for each gene
                for ti, tg in zip(test_idxs, test_genes):
                    if sub_adatac.obsm[predicted].columns[ti] != tg:
                        raise Warning ("ti not equal to tg: "+str(adata.var_names[ti])+" != "+str(tg))
                    ind_test_ci = np.nanmean(((adata[:,tg].X>prediction_sets[0][:,ti]) & (adata[:,tg].X<prediction_sets[1][:,ti]))[(G[:,ti]!=0)&(G_stdev[:,ti]!=0)&(adata[:,tg].X!=0)])
                    res_dict[method]['ind_gene_results'][tg]['test'].append(ind_test_ci)
                    
                del sub_adatac
                del prediction_sets
                del ind_test_ci
                gc.collect()

            # weighted average
            if i == 0:
                calibration_ci = np.array(calib_perc) * len(calib_genes)
                calibration_weight += len(calib_genes)
                test_ci = np.array(test_perc) * len(test_genes)
                test_weight += len(test_genes)
            else:
                calibration_ci += np.array(calib_perc) * len(calib_genes)
                calibration_weight += len(calib_genes)
                test_ci += np.array(test_perc) * len(test_genes)
                test_weight += len(test_genes)
                
            # Add new predictions
            for si, s_alpha in enumerate(save_alpha):
                conformalize_prediction_interval (sub_adata, predicted, calib_genes, alpha_level=s_alpha,
                                                  symmetric=symmetric, return_scores_dict=False)
                
                if i == 0:
                    if si == 0: # to avoid overwriting multiple times
                        adata.obsm[predicted] = pd.DataFrame(sub_adata.obsm[predicted][fold].values,
                                                          columns=fold,
                                                          index=adata.obs_names)
                    adata.obsm[predicted+f"_lo_{round((1-s_alpha)*100)}"] = pd.DataFrame(sub_adata.obsm[predicted+"_lo"][fold].values,
                                                      columns=fold,
                                                      index=adata.obs_names)
                    adata.obsm[predicted+f"_hi_{round((1-s_alpha)*100)}"] = pd.DataFrame(sub_adata.obsm[predicted+"_hi"][fold].values,
                                                      columns=fold,
                                                      index=adata.obs_names)
                else:
                    if si == 0: # to avoid overwriting multiple times
                        adata.obsm[predicted][fold] = sub_adata.obsm[predicted][fold].values.copy()
                    adata.obsm[predicted+f"_lo_{round((1-s_alpha)*100)}"][fold] = sub_adata.obsm[predicted+"_lo"][fold].values.copy()
                    adata.obsm[predicted+f"_hi_{round((1-s_alpha)*100)}"][fold] = sub_adata.obsm[predicted+"_hi"][fold].values.copy()
                
            del sub_adata
            gc.collect()

        # add results
        calibration_ci = calibration_ci / calibration_weight
        test_ci = test_ci / test_weight

        res_dict[method]['1-alpha'] = 1-alpha_levels
        res_dict[method]['calibration'] = calibration_ci
        res_dict[method]['test'] = test_ci
        
    return(res_dict, adata)


def measure_calibration_error (res_dict, key, method="average"):
    '''
    Scores the calibration results from the res_dict object (dictionary output of group_conformalize_from_intermediate())
    
    Parameters
    ----------
        res_dict [python dict]
        key [str] - key to access for scoring (i.e. the model name)
        method [str] = "average" or "gene" to report either the results on average calibration or average metric across all genes
        
    Returns
    -------
        score [float] - score for calibration error (lower is better)
    '''        
    from sklearn.metrics import auc
    
    if method == "gene":    
        auc_diffs = []
            
        for gene in res_dict[key]['ind_gene_results'].keys():
            diff = np.abs(res_dict[key]['ind_gene_results'][gene]['test'] - res_dict[key]['ind_gene_results'][gene]['1-alpha'])            
            auc_diffs.append(np.trapz(y=diff, x=res_dict[key]['ind_gene_results'][gene]['1-alpha']))
                
        score = np.nanmean(np.abs(auc_diffs))
        
    else:
        diff = np.abs(res_dict[key]['test'] - res_dict[key]['1-alpha'])            
        score = np.abs(np.trapz(y=diff, x=res_dict[key]['1-alpha']))
    
    return (score)


def group_multiple_imputation_testing_from_intermediate(dataset_name, methods, symmetric, condition, n_imputations=100,
                                                        group1=None, group2=None, savedir="SCPI", test="ttest"):
    '''
    Function for taking intermediate fold predictions and running multiple imputation t-tests
    
    Returns AnnData object with all test results saved in adata.var
    
    Parameters
    ----------
        dataset_name [str] - name of folder in DataUpload/
        methods [list of str] - list of method keys to use for prediction_sets
        symmetric [bool] - whether to use symmetric prediction intervals
        condition [str] - key in adata.obs to use for testing
        n_imputations [int] - number of multiple imputations
        group1 [None or str] - value in condition to use for group1 (if None, then will get results for all unique values)
        group2 [None or str] - value in condition to use for group2 (if None, then will use all other values as group2)
        savedir [str] - folder where the intermediate results are saved (independent folds)
        type_dataset [str] - default to "DataUpload" but may have additional options in the future
        
    Returns
    -------
        adata [AnnData] - anndata with testing results added to metadata
    ''' 
    # read in spatial data
    if os.path.isfile("DataUpload/"+dataset_name+"/Metadata.txt"):
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt",
                                   spatial_metadata = "DataUpload/"+dataset_name+"/Metadata.txt")
    else:
        adata = load_spatial_data("DataUpload/"+dataset_name+"/Spatial_count.txt",
                                  "DataUpload/"+dataset_name+"/Locations.txt")
    adata.var_names = [x.lower() for x in adata.var_names]
    
    for method in methods:

        dirpath = savedir+"/"+dataset_name+"_intermediate/"+method
        
        folds = np.load(os.path.join(savedir+"/"+dataset_name+"_intermediate/"+method,"folds.npy"), allow_pickle=True)

        # subset spatial data into shared genes
        gene_names = np.concatenate(folds)
        adata = adata[:, gene_names]

        for i, fold in enumerate(folds):

            # load adata within fold
            sub_adata = large_load(os.path.join(dirpath, "fold"+str(i)))
            target_genes = list(fold)

            # subset data
            predicted = method+"_predicted_expression"
            test_genes = target_genes.copy()
            calib_genes = [gene for gene in gene_names if gene not in test_genes]
            test_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in test_genes]
            calib_idxs = [np.where(sub_adata.obsm[predicted].columns==gene)[0][0] for gene in calib_genes]

            # run multiple imputation test
            keys_list = multiple_imputation_testing (sub_adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                                     group1=group1, group2=group2, symmetric=symmetric, return_keys=True, test=test)
            
            if i == 0:
                for key in keys_list:
                    adata.var[key] = np.zeros(adata.shape[1])
                    
                adata.obsm[predicted] = pd.DataFrame(sub_adata.obsm[predicted][fold].values,
                                                  columns=fold,
                                                  index=adata.obs_names)
            for key in keys_list:
                adata.var[key][[np.where(adata.var_names==gene)[0][0] for gene in fold]] = sub_adata.uns[key][fold].values.flatten()
                adata.obsm[predicted][fold] = sub_adata.obsm[predicted][fold].values.copy()
                
    return(adata)