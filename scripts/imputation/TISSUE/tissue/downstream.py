# Contains functions for all downstream applications of TISSUE calibration scores and prediction intervals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import anndata as ad
import os
import sys

#from tissue.main import build_calibration_scores, get_spatial_uncertainty_scores_from_metadata
from .main import build_calibration_scores, get_spatial_uncertainty_scores_from_metadata


def multiple_imputation_testing (adata, predicted, calib_genes, condition, test="ttest", n_imputations=100,
                                 group1=None, group2=None, symmetric=False, return_keys=False, save_mi=False):
    '''
    Uses multiple imputation with the score distributions to perform hypothesis testing
    
    Parameters
    ----------
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm that corresponds to predicted gene expression
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        condition [str] - key in adata.obs for which to compute the hypothesis test
            group1 [value] - value in adata.obs[condition] identifying the first comparison group
                             if None, will perform group vs all comparisons for all unique values in adata.obs[condition]
            group2 [value] - value in adata.obs[condition] identifying the second comparison group
                             if None, will compare against all values that are not group1
        test [str] - statistical test to use:
                        "ttest" - two-sample t-test using Rubin's rules (best theoretical support/guarantee)
                        "wilcoxon_greater" - one-sample wilcoxon (Mann-Whitney U test) for greater expression using p-value transformation
                        "wilcoxon_less" - one-sample wilcoxon (Mann-Whitney U test) for lesser expression using p-value transformation
                        "spatialde" - SpatialDE test using p-value transformation
        n_imputations [int] - number of imputations to use
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        return_keys [bool] - whether to return the keys for which to access the results from adata
        save_mi [False or str] - multiple imputation saving (only used for multiple_imputation_ttest())
        
    Returns
    -------
        Modifies adata in-place to add the statistics and test results to metadata
        Optionally returns the keys to access the results from adata
        
    '''
    #####################################################################
    # T-test (default) - this is the option with best theoretical support
    #####################################################################
    if test == "ttest":
        keys = multiple_imputation_ttest (adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                 group1=group1, group2=group2, symmetric=symmetric, save_mi=save_mi)
            
    #####################################################################
    # One-sample ("less"/"greater") Wilcoxon test  
    #####################################################################    
    elif test == "wilcoxon_less":
        keys = multiple_imputation_wilcoxon (adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                 group1=group1, group2=group2, symmetric=symmetric, direction='less')
    elif test == "wilcoxon_greater":
        keys = multiple_imputation_wilcoxon (adata, predicted, calib_genes, condition, n_imputations=n_imputations,
                                 group1=group1, group2=group2, symmetric=symmetric, direction='greater')
                                 
    #####################################################################
    # SpatialDE (spatially variable genes) test 
    ##################################################################### 
    elif test == "spatialde":
        keys = multiple_imputation_spatialde (adata, predicted, calib_genes, n_imputations=n_imputations, symmetric=symmetric)
    
    # raise exception if test does not match options
    else:
        raise Exception ("Specified test not recognized")
        
    if return_keys is True:
        
        return(keys)


def multiple_imputation_spatialde (adata, predicted, calib_genes, n_imputations=100, symmetric=False):
    '''
    Runs TISSUE multiple imputation SpatialDE test using p-value transformation
    
    See multiple_imputation_testing() for details on parameters
    '''
    import SpatialDE
    
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                     include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    
    ### Multiple imputation

    # init dictionary to hold results
    stat_dict = {}
    stat_dict["pvalue"] = {}
    
    for m in range(n_imputations):
        
        # generate new imputation
        new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
        
        key = "spatialde"
    
        if m == 0: # init list
            stat_dict["pvalue"][key] = []
        
        # get spatialDE p-values
        normalized_matrix = new_G/(1+np.sum(new_G,axis=1)[:,None])
        normalized_matrix = np.log1p((normalized_matrix-np.min(normalized_matrix)) * 100) 
        sp_df = pd.DataFrame(normalized_matrix,
                          columns=adata.obsm[predicted].columns,
                          index=adata.obsm[predicted].index)

        results = SpatialDE.run(adata.obsm['spatial'], sp_df)

        # sort by gene name order
        results.drop_duplicates(subset = ['g'], keep = 'first', inplace = True) # workaround duplication SpatialDE bug
        results.g = results.g.astype("category")
        results.g = results.g.cat.set_categories(adata.obsm[predicted].columns)
        results = results.sort_values(["g"])

        # get pvalues
        pval = list(results["pval"])
        stat_dict["pvalue"][key].append(pval)

    # pool statistics
    pooled_results_dict = {}
    pooled_results_dict['pvalue'] = {}
    # for each test grouping
    for key in stat_dict['pvalue'].keys():
        pooled_results_dict['pvalue'][key] = []
        pval_arr = np.vstack(stat_dict['pvalue'][key])
        # for each gene, get mi pvalue
        for ci in range(pval_arr.shape[1]):
            mi_pval = multiply_imputed_pvalue (pval_arr[:,ci], method="licht_rubin")
            pooled_results_dict['pvalue'][key].append(mi_pval)
     
    # add stats to adata
    keys_list = []
    for key_measure in pooled_results_dict.keys():
        for key_comparison in pooled_results_dict[key_measure].keys():
            adata.uns[predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure] = pd.DataFrame(np.array(pooled_results_dict[key_measure][key_comparison])[None,:],
                                                                                                 columns=adata.obsm[predicted].columns)
            keys_list.append(predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure)
    
    return(keys_list)


def multiple_imputation_wilcoxon (adata, predicted, calib_genes, condition, n_imputations=100,
                                  group1=None, group2=None, symmetric=False, direction="greater"):
    '''
    Runs TISSUE multiple imputation one-sample Wilcoxon (greater/lesser) test using p-value transformation
    
    See multiple_imputation_testing() for details on parameters
    '''
    from scipy.stats import mannwhitneyu
    
    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                     include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    
    ### Multiple imputation

    # init dictionary to hold results
    stat_dict = {}
    stat_dict["pvalue"] = {}
    
    # cast condition to str
    condition = str(condition)
    
    for m in range(n_imputations):
        
        # generate new imputation
        new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
            
        if group1 is None: # pairwise comparisons against all
            
            for g1 in np.unique(adata.obs[condition]):
                
                key = str(g1)+"_all"
            
                if m == 0: # init list
                    stat_dict["pvalue"][key] = []
                
                g1_bool = (adata.obs[condition] == g1) # g1
                g2_bool = (adata.obs[condition] != g1) # all other
                
                # get SpatialDE p-values
                pval = []
                for ci in range(new_G.shape[1]):
                    u,p = mannwhitneyu(new_G[g1_bool,ci], new_G[g2_bool,ci], alternative=direction)
                    pval.append(p)
                
                stat_dict["pvalue"][key].append(pval)
                
        elif group2 is None: # group1 vs all
        
            key = str(group1)+"_all"
            
            if m == 0: # init list
                stat_dict["pvalue"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] != group1) # all other
            
            # get wilcoxon p-values
            pval = []
            for ci in range(new_G.shape[1]):
                u,p = mannwhitneyu(new_G[g1_bool,ci], new_G[g2_bool,ci], alternative=direction)
                pval.append(p)
            
            stat_dict["pvalue"][key].append(pval)
            
        else: # group1 vs group2
            
            key = str(group1)+"_"+str(group2)
            
            if m == 0: # init list
                stat_dict["pvalue"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] == group2) # g2
            
            # get wilcoxon p-values
            pval = []
            for ci in range(new_G.shape[1]):
                u,p = mannwhitneyu(new_G[g1_bool,ci], new_G[g2_bool,ci], alternative=direction)
                pval.append(p)
                
            stat_dict["pvalue"][key].append(pval)

    # pool statistics
    pooled_results_dict = {}
    pooled_results_dict['pvalue'] = {}
    # for each test grouping
    for key in stat_dict['pvalue'].keys():
        pooled_results_dict['pvalue'][key] = []
        pval_arr = np.vstack(stat_dict['pvalue'][key])
        # for each gene, get mi pvalue
        for ci in range(pval_arr.shape[1]):
            mi_pval = multiply_imputed_pvalue (pval_arr[:,ci], method="licht_rubin")
            pooled_results_dict['pvalue'][key].append(mi_pval)
     
    # add stats to adata
    keys_list = []
    for key_measure in pooled_results_dict.keys():
        for key_comparison in pooled_results_dict[key_measure].keys():
            adata.uns[predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure] = pd.DataFrame(np.array(pooled_results_dict[key_measure][key_comparison])[None,:],
                                                                                                 columns=adata.obsm[predicted].columns)
            keys_list.append(predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure)
    
    return(keys_list)


def multiply_imputed_pvalue (pvalues, method="licht_rubin"):
    '''
    Computes a multiply imputed p-value from a list of p-values according to Licht-Rubin procedure or median procedure
    
    Parameters
    ----------
        pvalues [array-like] - array of p-values from multiple imputation tests
        method [str] - which method for p-value calculation to use: "licht_rubin" or "median"
        
    Returns
    -------
        mi_pvalue [float] - p-value modified for multiple imputation
        
    See reference for technical details: https://stefvanbuuren.name/fimd/sec-multiparameter.html#sec:chi
    '''
    from scipy.stats import norm
    
    if method == "licht_rubin":
        z = norm.ppf(pvalues)  # transform to z-scale
        num = np.nanmean(z)
        den = np.sqrt(1 + np.nanvar(z))
        mi_pvalue = norm.cdf( num / den) # average and transform back
    
    elif method == "median":
        mi_pvalue = np.nanmedian(pvalues)
    
    else:
        raise Exception ("method for multiply_imputed_pvalue() not recognized")

    return(mi_pvalue)



def multiple_imputation_ttest (adata, predicted, calib_genes, condition, n_imputations=100,
                               group1=None, group2=None, symmetric=False, save_mi=False):
    '''
    Runs TISSUE multiple imputation two-sample t-test using Rubin's rules
    
    See multiple_imputation_testing() for details on parameters
    
    Additional Parameters
    ---------------------
        save_mi [False or str] - if not False, then saves "multiple_imputations.npy" stacked matrix of imputed gene expression at save_mi path
    '''

    # get uncertainties and scores from saved adata
    scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    ### Building calibration sets for scores
    
    scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                     include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    
    ### Multiple imputation

    # init dictionary to hold results (for independent two-sample t-test)
    stat_dict = {}
    stat_dict["mean_difference"] = {}
    stat_dict["standard_deviation"] = {}
    
    # cast condition to str
    condition = str(condition)
    
    new_G_list = [] # for saving multiple imputations
    
    for m in range(n_imputations):
        
        # generate new imputation
        new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
        if save_mi is not False:
            new_G_list.append(new_G)
    
        # calculate statistics for the imputation using approach from Palmer & Peer, 2016
        
        if group1 is None: # pairwise comparisons against all
            
            for g1 in np.unique(adata.obs[condition]):
                
                key = str(g1)+"_all"
            
                if m == 0: # init list
                    stat_dict["mean_difference"][key] = []
                    stat_dict["standard_deviation"][key] = []
                
                g1_bool = (adata.obs[condition] == g1) # g1
                g2_bool = (adata.obs[condition] != g1) # all other
                
                mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
                stat_dict["mean_difference"][key].append(mean_diff)
                stat_dict["standard_deviation"][key].append(pooled_sd)
                
        elif group2 is None: # group1 vs all
        
            key = str(group1)+"_all"
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] != group1) # all other
            
            mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)
            
        else: # group1 vs group2
            
            key = str(group1)+"_"+str(group2)
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] == group2) # g2
            
            mean_diff, pooled_sd = get_ttest_stats(new_G, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)

    # pool statistics and perform t-test
    pooled_results_dict = pool_multiple_stats(stat_dict)
     
    # add stats to adata
    keys_list = []
    for key_measure in pooled_results_dict.keys():
        for key_comparison in pooled_results_dict[key_measure].keys():
            adata.uns[predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure] = pd.DataFrame(pooled_results_dict[key_measure][key_comparison][None,:],
                                                                                                 columns=adata.obsm[predicted].columns)
            keys_list.append(predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure)
    
    # save multiple imputations
    if save_mi is not False:
        # stack all imputations and save
        stacked_mi = np.dstack(new_G_list)
        np.save(os.path.join(save_mi,f"{predicted}.npy"), stacked_mi)
    
    return(keys_list)



# def multiple_imputation_save (sig_dirpath, adata, predicted, calib_genes, condition, n_imputations=100,
                                 # group1=None, group2=None, symmetric=False, return_keys=False, load_mi=False):
    # '''
    # Parameters
    # ----------
        # sig_dirpath [str] - path to save the "multiple_imputations.npy" stacked matrix object
        # adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        # predicted [str] - key in adata.obsm that corresponds to predicted gene expression
        # calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        # n_imputations [int] - number of imputations to use
        # symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        
    # Returns
    # -------
        # Saves stacked matrix of multiple imputations in "multiple_imputations.npy" for loading with multiple_imputation_gene_signature()
    # '''
    # # get uncertainties and scores from saved adata
    # scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
    
    # ### Building calibration sets for scores
    
    # scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                     # include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    
    # ### Multiple imputation
    
    # new_G_list = []
    
    # for m in range(n_imputations):
        
        # # generate new imputation
        # new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
        # new_G_list.append(new_G)
        
    # # stack all imputations and save
    # stacked_mi = np.dstack(new_G_list)
    # np.save(os.path.join(sig_dirpath,"multiple_imputations.npy"), stacked_mi)


def multiple_imputation_gene_signature (sig_dirpath, adata, predicted, calib_genes, condition, n_imputations=100,
                                 group1=None, group2=None, symmetric=False, return_keys=False, load_mi=False):
    '''
    Uses multiple imputation with the score distributions to perform hypothesis testing on gene signatures
    
    Parameters
    ----------
        sig_dirpath [str] - path to the directory containing the gene signatures organized as:
                            sig_dirpath/
                                {name of signature 1}/
                                {name of signature N}/
                                    genes.txt - text file with each row being a gene name
                                    coefficients.txt - optional text file with each row being a float weight for corresponding gene
        adata [AnnData] - contains adata.obsm[predicted] corresponding to the predicted gene expression
        predicted [str] - key in adata.obsm that corresponds to predicted gene expression
        calib_genes [list or arr of str] - names of the genes in adata.var_names that are used in the calibration set
        condition [str] - key in adata.obs for which to compute the hypothesis test
            group1 [value] - value in adata.obs[condition] identifying the first comparison group
                             if None, will perform group vs all comparisons for all unique values in adata.obs[condition]
            group2 [value] - value in adata.obs[condition] identifying the second comparison group
                             if None, will compare against all values that are not group1
        n_imputations [int] - number of imputations to use
        symmetric [bool] - whether to have symmetric (or non-symmetric) prediction intervals
        return_keys [bool] - whether to return the keys for which to access the results from adata
        load_mi [bool] - whether to save "{predicted}.npy" stacked matrix of all multiple imputations at sig_dirpath 
        
    Returns
    -------
        Modifies adata in-place to add the statistics and test results to metadata
        Optionally returns the keys to access the results from adata
        
    '''
    #####################################################################
    # T-test (default) - this is the only option currently for signatures
    #####################################################################
    
    if load_mi is False:
        # get uncertainties and scores from saved adata
        scores, residuals, G_stdev, G, groups = get_spatial_uncertainty_scores_from_metadata (adata, predicted)
        
        ### Building calibration sets for scores
        
        scores_flattened_dict = build_calibration_scores(adata, predicted, calib_genes, symmetric=symmetric,
                                                         include_zero_scores=True, trim_quantiles=[None, 0.8]) # trim top 20% scores
    else: # load in saved multiple imputations
        mi_path = os.path.join(sig_dirpath,f"{predicted}.npy") # path to saved multiple imputations
        mi_stacked = np.load(mi_path)
    
    ### Multiple imputation

    # init dictionary to hold results (for independent two-sample t-test)
    stat_dict = {}
    stat_dict["mean_difference"] = {}
    stat_dict["standard_deviation"] = {}
    
    # cast condition to str
    condition = str(condition)
    
    for m in range(n_imputations):
        
        # generate new imputation
        if load_mi is False:
            new_G = sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=symmetric)
        else:
            new_G = mi_stacked[:,:,m].copy() # take the m-th multiple imputation
        
        # compute all signatures
        imputed_sigs = [] 
        sig_names = []
        
        for sigdir in next(os.walk(sig_dirpath))[1]: # iterate all top-level signature directories
            # read in genes
            with open(os.path.join(sig_dirpath,sigdir,"genes.txt")) as f:
                signature_genes = [line.rstrip() for line in f]
            signature_genes = np.array([x.lower() for x in signature_genes])
            # load coefficients (if any)
            if os.path.isfile(os.path.join(sig_dirpath,sigdir,"coefficients.txt")):
                signature_coefficients = np.loadtxt(os.path.join(sig_dirpath,sigdir,"coefficients.txt"))
            else:
                signature_coefficients = np.ones(len(signature_genes))
            # subset into shared genes
            shared_gene_idxs = [ii for ii in range(len(signature_genes)) if signature_genes[ii] in adata.obsm[predicted].columns]
            signature_genes = signature_genes[shared_gene_idxs]
            signature_coefficients = signature_coefficients[shared_gene_idxs]
            # if non-empty signature, then compute
            if len(signature_genes) > 0:
                # compute signature
                subset_new_G = pd.DataFrame(new_G, columns = adata.obsm[predicted].columns)[signature_genes].values
                sig_value = np.nansum(subset_new_G*signature_coefficients, axis=1)
                # append signature value and name
                imputed_sigs.append(sig_value)
                sig_names.append(sigdir)

        # construct gene signature matrix
        imputed_sigs = np.vstack(imputed_sigs).T
        
        # keep running average of imputed gene signatures
        if m == 0:
            mean_imputed_sigs = imputed_sigs * 1/n_imputations
        else:
            mean_imputed_sigs += imputed_sigs * 1/n_imputations
        
        # calculate statistics for the imputation using approach from Palmer & Peer, 2016
        
        if group1 is None: # pairwise comparisons against all
            
            for g1 in np.unique(adata.obs[condition]):
                
                key = str(g1)+"_all"
            
                if m == 0: # init list
                    stat_dict["mean_difference"][key] = []
                    stat_dict["standard_deviation"][key] = []
                
                g1_bool = (adata.obs[condition] == g1) # g1
                g2_bool = (adata.obs[condition] != g1) # all other
                
                mean_diff, pooled_sd = get_ttest_stats(imputed_sigs, g1_bool, g2_bool) # get ttest stats
                stat_dict["mean_difference"][key].append(mean_diff)
                stat_dict["standard_deviation"][key].append(pooled_sd)
                
        elif group2 is None: # group1 vs all
        
            key = str(group1)+"_all"
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] != group1) # all other
            
            mean_diff, pooled_sd = get_ttest_stats(imputed_sigs, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)
            
        else: # group1 vs group2
            
            key = str(group1)+"_"+str(group2)
            
            if m == 0: # init list
                stat_dict["mean_difference"][key] = []
                stat_dict["standard_deviation"][key] = []
            
            g1_bool = (adata.obs[condition] == group1) # g1
            g2_bool = (adata.obs[condition] == group2) # g2
            
            mean_diff, pooled_sd = get_ttest_stats(imputed_sigs, g1_bool, g2_bool) # get ttest stats
            stat_dict["mean_difference"][key].append(mean_diff)
            stat_dict["standard_deviation"][key].append(pooled_sd)

    # pool statistics and perform t-test
    pooled_results_dict = pool_multiple_stats(stat_dict)
     
    # add stats to adata
    keys_list = []
    for key_measure in pooled_results_dict.keys():
        for key_comparison in pooled_results_dict[key_measure].keys():
            adata.uns[predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure] = pd.DataFrame(pooled_results_dict[key_measure][key_comparison][None,:],
                                                                                                 columns=sig_names)
            keys_list.append(predicted.split("_")[0]+"_"+key_comparison+"_"+key_measure)
                    
    # add gene sigs to adata
    adata.obsm[predicted+"_gene_signatures"] = pd.DataFrame(mean_imputed_sigs, columns=sig_names, index=adata.obs_names)
    
    if return_keys is True:
        
        return(keys_list)



def sample_new_imputation_from_scores (G, G_stdev, groups, scores_flattened_dict, symmetric=False):
    '''
    Creates a new imputation by sampling from scores and adding to G
    
    Parameters
    ----------
        G, G_stdev, groups - outputs of get_spatial_uncertainty_scores_from_metadata()
        scores_flattened_dict - output of build_calibration_scores()
    
    See multiple_imputation_testing() for more details of arguments
    
    Returns
    -------
        new_G - array of the new sampled predicted gene expression (same dimensions as new_G: cells x genes)
    '''
    new_scores = np.zeros(G.shape) # init array for sampled scores
    new_add_sub = np.zeros(G.shape) # init array for add/subtract coefs
    
    # for each group, sample calibration score and corresponding imputations
    unique_groups, unique_counts = np.unique(groups[~np.isnan(groups)], return_counts=True)
    
    for ui, group in enumerate(unique_groups):
        count = unique_counts[ui] # get number of values in group
        
        # sample scores and add/sub indicators
        if symmetric is True:
            scores_flattened = scores_flattened_dict[str(group)] # get scores
            if len(scores_flattened) < 100: # default to full set if <100 in group
                scores_flattened = scores_flattened_dict[str(np.nan)]
            sampled_scores = np.random.choice(scores_flattened, count, replace=True) # with replacement, sample scores
            add_sub = np.random.choice([-1,1], count, replace=True) # add or subtract
        else:
            scores_lo_flattened = scores_flattened_dict[str(group)][0]
            scores_hi_flattened = scores_flattened_dict[str(group)][1]
            if (len(scores_lo_flattened) < 100) or (len(scores_hi_flattened) < 100): # default to full set if <100 in group
                scores_lo_flattened = scores_flattened_dict[str(np.nan)][0]
                scores_hi_flattened = scores_flattened_dict[str(np.nan)][1]
            scores_flattened = np.concatenate((scores_lo_flattened, scores_hi_flattened))
            lo_hi_indicators = np.concatenate(([-1]*len(scores_lo_flattened), [1]*len(scores_hi_flattened)))
            # sample indices
            sampled_idxs = np.random.choice(np.arange(len(scores_flattened)), count, replace=True) # with replacement
            sampled_scores = scores_flattened[sampled_idxs]
            add_sub = lo_hi_indicators[sampled_idxs]
        
        # append to new_scores and new_add_sub
        new_scores[groups==group] = sampled_scores
        new_add_sub[groups==group] = add_sub
        
    # calculate new imputation
    new_G = G + new_add_sub*(new_scores*G_stdev)

    return (new_G)


def get_ttest_stats(G, g1_bool, g2_bool):
    '''
    Computes mean_diff and pooled SD for each column of G independently
    
    Parameters
    ----------
        G [array] - 2D array with columns as genes and rows as cells
        g1_bool [bool array] - 1D array with length equal to number of rows in G; labels group1
        g2_bool [bool array] - 1D array with length equal to number of rows in G; labels group2
        
    Returns
    -------
        mean_diff - mean difference for t-test
        pooled_sd - pooled standard deviation for t-test
    '''
    mean_diff = np.nanmean(G[g1_bool,:], axis=0) - np.nanmean(G[g2_bool,:], axis=0)
    n1 = np.count_nonzero(~np.isnan(G[g1_bool,:]), axis=0)
    n2 = np.count_nonzero(~np.isnan(G[g2_bool,:]), axis=0)
    sp = np.sqrt( ( (n1-1)*(np.nanvar(G[g1_bool,:],axis=0)) + (n2-1)*(np.nanvar(G[g2_bool,:],axis=0)) ) / (n1+n2-2) )
    pooled_sd = np.sqrt(1/n1 + 1/n2) * sp
    
    return(mean_diff, pooled_sd)


def two_sample_ttest (G, g1_bool, g2_bool):
    '''
    Computes two-sample t-test for unequal sample sizes using get_ttest_stats()
    
    Parameters
    ----------
        G [array] - 2D array with columns as genes and rows as cells
        g1_bool [bool array] - 1D array with length equal to number of rows in G; labels group1
        g2_bool [bool array] - 1D array with length equal to number of rows in G; labels group2
        
    Returns
    -------
        tt - t-statistic
        pp - p-value
    '''
    from scipy import stats
    # calculate t-stat
    mean_diff, pooled_sd = get_ttest_stats(G, g1_bool, g2_bool)
    tt = mean_diff/pooled_sd
    # calculate dof
    n1 = np.count_nonzero(~np.isnan(G[g1_bool,:]), axis=0)
    n2 = np.count_nonzero(~np.isnan(G[g2_bool,:]), axis=0)
    dof = n1+n2-2
    # calculate p-value
    pp = 2*(1 - stats.t.cdf(np.abs(tt), dof))
    
    return(tt, pp)


def pool_multiple_stats(stat_dict):
    '''
    Pool stats across multiple imputations for t-test
    
    Parameters
    ----------
        stat_dict [dict] - dictionary containing statistical testing results (generated in multiple_imputation_ttest())
        
    Returns
    -------
        results_dict [dict] - dictionary containing the pooled statistics from using Rubin's rules
    '''
    from scipy import stats
    
    # init results_dict
    results_dict = {}
    results_dict["tstat"] = {}
    results_dict["pvalue"] = {}
    
    results_dict["varw"] = {}
    results_dict["varb"] = {}
    results_dict["poolmean"] = {}
    
    for key in stat_dict["mean_difference"].keys():
        
        d = len(stat_dict["mean_difference"][key])
        
        # compute pooled terms
        pooled_mean = np.mean(np.vstack(stat_dict["mean_difference"][key]), axis=0)
        var_w = np.mean(np.vstack(stat_dict["standard_deviation"][key])**2, axis=0) # within-draw sample variance
        var_b = 1/(d-1) * np.sum((np.vstack(stat_dict["mean_difference"][key])-pooled_mean)**2, axis=0) # between-draw sample variance
        var_MI = var_w + (1+1/d)*var_b # multiple imputation variance
        
        test_stat = pooled_mean / np.sqrt(var_MI) # pooled t statistic
        
        # compute pvalue from T distribution
        dof = (d-1)*(1+(d*var_w)/((d+1)*var_b))**2 # degrees of freedom for T distribution
        pval = 2*(1 - stats.t.cdf(np.abs(test_stat), dof))
        
        # Add test statistic and pvalue
        results_dict["tstat"][key] = test_stat
        results_dict["pvalue"][key] = pval
        
        # Add intermediate stats (for debugging, etc)
        results_dict["varw"][key] = var_w
        results_dict["varb"][key] = var_b
        results_dict["poolmean"][key] = pooled_mean
    
    return(results_dict)



def weighted_PCA(adata, imp_method, pca_method="wpca", weighting="inverse_norm_pi_width", quantile_cutoff=None,
                 n_components=15, replace_inf=None, binarize=0.2, binarize_ratio=10, log_transform=False,
                 scale=True, tag="", return_weights=False,):
    '''
    Runs weighted PCA using the "wpca" package: https://github.com/jakevdp/wpca
    
    Parameters
    ----------
        adata [AnnData] - should be the AnnData after running conformalize_prediction_interval()
                        - must include in obsm: {imp_method}_predicted_expression,
                                                {imp_method}_predicted_expression_lo,
                                                {imp_method}_predicted_expression_hi
        imp_method [str] - specifies which imputation method to return PCA for (e.g. 'knn', 'spage', 'tangram')
        pca_method [str] - "wpca" for WPCA (Delchambre, 2014), "empca" for EMPCA (Bailey, 2012), "pca" for PCA
        weighting [str] - "uniform" (regular PCA)
                          "inverse_pi_width" (weights are 1/(prediction interval width))
                          "inverse_norm_pi_width" (weights are predicted expression/(prediction interval width))
        quantile_cutoff [None or float] - quantile (between 0 and 1) for which to set a ceiling for the weights
        n_components [int] - number of principal components
        replace_inf [None, str, float] - what to replace np.inf with (after all other weight transforms); if None, keeps np.inf
                                         can also be "max" or "min" to replace with the max or min weights
        binarize [bool] - binarizes the weights with Otsu threshold -- if larger than threshold, set to 1; else 1e-2
        binarize_ratio [int or float] - how much to "upweight" values greater than the binarized threshold
        log_transform [bool] - whether to log1p transform weights (will be done before binarization if binarize=True)
        scale [bool - whether to scale data with StandardScaler() before running WPCA
        tag [str] - additional tag to append to the obsm key for storing the PCs
        return_weights [bool] - whether to return weights used in WPCA
     
    Returns
    -------
        Stores the result in adata.obsm["{imp_method}_predicted_expression_PC{n_components}_{tag}"]
        Optionally returns the array of weights used in WPCA
    
    Refer to postprocess_weights() for order for weight calculations
    '''
    from wpca import PCA, WPCA, EMPCA
    
    predicted = f"{imp_method}_predicted_expression"
    
    # get gene names/order
    genes = adata.obsm[predicted].columns
    
    # determine weights
    if weighting == "inverse_pi_width":
        weights = 1/(adata.obsm[predicted+'_hi'][genes].values-adata.obsm[predicted+'_lo'][genes].values)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "inverse_norm_pi_width":
        weights = 1/(adata.obsm[predicted+'_hi'][genes].values-adata.obsm[predicted+'_lo'][genes].values)
        weights = weights / np.nanmean(weights, axis=0)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "uniform":
        weights = np.ones(adata.obsm[predicted].shape)
    elif weighting == "inverse_residual":
        weights = 1/np.abs(adata.obsm[predicted][genes].values - np.array(adata[:,genes].X))
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    elif weighting == "inverse_norm_residual":
        weights = 1/np.abs(adata.obsm[predicted][genes].values - np.array(adata[:,genes].X))
        weights = weights / np.nanmean(weights, axis=0)
        weights = postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform)
    else:
        raise Exception("weighting not recognized")
    
    # scaling
    if scale is True:
        X = StandardScaler().fit_transform(adata.obsm[predicted].values)
    else:
        X = adata.obsm[predicted].values
    
    # run weighted PCA
    if pca_method == "wpca":
        X_red = WPCA(n_components=n_components).fit_transform(X, weights=weights)
    elif pca_method == "empca":
        X_red = EMPCA(n_components=n_components).fit_transform(X, weights=weights)
    elif pca_method == "pca":
        X_red = PCA(n_components=n_components).fit_transform(X)
    elif pca_method == "gwpca": # gene-weighted PCA
        weights = np.nanmean(weights, axis=0)
        X_red = PCA(n_components=n_components).fit_transform(X * weights)
    else:
        raise Exception("pca_method not recognized")
        
    # add PCs to adata
    adata.obsm[predicted+f"_PC{n_components}_{tag}"] = X_red

    if return_weights is True:
        return(weights)


def postprocess_weights(weights, quantile_cutoff, replace_inf, binarize, binarize_ratio, log_transform):
    '''
    Method for postprocessing weights (filter with cutoff, replace inf, etc) for weighted_PCA()
    
    Refer to weighted_pca() for details on arguments
    '''
    # cutoff weights
    if quantile_cutoff is not None:
        cutoff = np.nanquantile(weights, quantile_cutoff)
        weights[np.isfinite(weights) & (weights >= cutoff)] = cutoff
    
    # log-transform
    if log_transform is True:
        weights = np.log1p(weights)
    
    # binarize weights
    if binarize is True:
        from skimage.filters import threshold_otsu
        cutoff = threshold_otsu(weights[np.isfinite(weights)])
        weights[np.isfinite(weights) & (weights >= cutoff)] = 1
        weights[np.isfinite(weights) & (weights < cutoff)] = 1/binarize_ratio
    elif binarize is False:
        pass
    elif isinstance(binarize, float) or isinstance(binarize, int):
        cutoff = np.nanquantile(weights, binarize)
        weights[np.isfinite(weights) & (weights >= cutoff)] = 1
        weights[np.isfinite(weights) & (weights < cutoff)] = 1/binarize_ratio
        
    # deal with infs (from division by zero)
    if replace_inf == "max":
        weights[~np.isfinite(weights)] = np.nanmax(weights[np.isfinite(weights)])
    elif replace_inf == "min":
        weights[~np.isfinite(weights)] = np.nanmin(weights[np.isfinite(weights)])
    elif replace_inf == "mean":
        weights[~np.isfinite(weights)] = np.nanmean(weights[np.isfinite(weights)])
    elif replace_inf == "median":
        weights[~np.isfinite(weights)] = np.nanmedian(weights[np.isfinite(weights)])
    elif isinstance(replace_inf, float) or isinstance(replace_inf, int):
        weights[~np.isfinite(weights)] = replace_inf
    
    return(weights)


def filtered_PCA(adata, imp_method, proportion=0.05, stratification=None, n_components=15, scale=True, normalize=False,
                 tag="", return_keep_idxs=False):
    '''
    Runs filtered PCA using the TISSUE cell filtering approach
    
    Parameters
    ----------
        adata [AnnData] - should be the AnnData after running conformalize_prediction_interval()
                        - must include in obsm: {imp_method}_predicted_expression,
                                                {imp_method}_predicted_expression_lo,
                                                {imp_method}_predicted_expression_hi
        imp_method [str] - specifies which imputation method to return PCA for (e.g. 'knn', 'spage', 'tangram')
        proportion [float] - between 0 and 1; proportion of most uncertain cells to drop
        stratification [None or 1d numpy array] - array of values to stratify the drop by
                                                - same length as number of rows in X
                                                - if None, no stratification
        n_components [int] - number of principal components
        scale [bool] - whether to scale data with StandardScaler() before running PCA
        normalize [bool] - whether to normalize prediction interval width by the absolute predicted expression value
        tag [str] - additional tag to append to the obsm key for storing the PCs
        return_keep_idxs [bool] - whether to return the keep_idxs for filtering
    
    Returns
    -------
        Stores the result in adata.obsm["{imp_method}_predicted_expression_PC{n_components}_{tag}"]
        Optionally returns the indices corresponding to the observations to keep after filtering
    '''    
    predicted = f"{imp_method}_predicted_expression"
    
    # get predicted expression matrices
    X = adata.obsm[predicted].values.copy()
    
    # get uncertainty (PI width) for filtering
    X_uncertainty = adata.obsm[f'{predicted}_hi'].values - adata.obsm[f'{predicted}_lo'].values
    if normalize is True:
        X_uncertainty = X_uncertainty / (1+np.abs(adata.obsm[f'{predicted}'].values))
    
    # filter cells
    keep_idxs = detect_uncertain_cells(X_uncertainty, proportion=proportion, stratification=stratification)
    X_filtered = X[keep_idxs,:].copy()
    
    # scaling
    if scale is True:
        scaler = StandardScaler().fit(X_filtered)
        X = scaler.transform(X)
        X_filtered = scaler.transform(X_filtered)
    
    # run PCA
    pca = PCA(n_components=n_components).fit(X_filtered)
    X_red = pca.transform(X)
    X_red_filtered = pca.transform(X_filtered)
        
    # add PCs to adata
    adata.obsm[predicted+f"_PC{n_components}_{tag}"] = X_red
    adata.uns[predicted+f"_PC{n_components}_filtered_{tag}"] = X_red_filtered
    
    if return_keep_idxs is True:
        return (keep_idxs)



def detect_uncertain_cells (X, proportion=0.05, stratification=None):
    '''
    Method for dropping a portion of the most uncertain cells from the input. 
    
    Parameters
    ----------
        X [2d numpy array] - array of uncertainty values 
        proportion [float] - between 0 and 1; proportion of most uncertain cells to drop
        stratification [None or 1d numpy array] - array of values to stratify the drop by
                                                - same length as number of rows in X
                                                - if None, no stratification
        
    Returns
    -------
        keep_idxs [list] - array of row indices after dropping most uncertain cells
    '''
    from scipy.stats import zscore
    
    if stratification is not None: # drop cells within each strata independently
    
        drop_idxs = []
        
        for strata in np.unique(stratification):
            
            # compute scores
            X_strat = X[stratification==strata,:].copy() # calc gene z-scores
            orig_idxs = np.arange(X.shape[0])[stratification==strata]
            cell_scores = np.nanmean(zscore(X_strat, axis=0), axis=1) # average z-score for each cell
            
            # determine cutoff score and indices to drop
            if (isinstance(proportion, float)) or (isinstance(proportion, int)):
                cutoff_idx = int(np.ceil(proportion*len(cell_scores))) # number of cells to drop
                strata_drop_idxs = np.argsort(cell_scores)[::-1][:cutoff_idx]
            elif proportion == "otsu":
                from skimage.filters import threshold_otsu
                cutoff = threshold_otsu(cell_scores)
                strata_drop_idxs = [i for i in range(len(cell_scores)) if cell_scores[i] > cutoff]
            else:
                raise Exception("proportion specified not valid")
                
            drop_idxs.append(orig_idxs[strata_drop_idxs]) # get idxs of highest scores
            
        drop_idxs = list(np.concatenate(drop_idxs))
    
    else:
        
        # compute scores
        cell_scores = zscore(X, axis=0).mean(axis=1) # average z-score for each cell
        
        # determine cutoff score and indices to drop
        if (isinstance(proportion, float)) or (isinstance(proportion, int)):
            cutoff_idx = int(np.ceil(proportion*len(cell_scores))) # number of cells to drop
            drop_idxs = list(np.argsort(cell_scores)[::-1][:cutoff_idx]) # get idxs of highest scores
        elif proportion == "otsu":
            from skimage.filters import threshold_otsu
            cutoff = threshold_otsu(cell_scores)
            drop_idxs = [i for i in range(len(cell_scores)) if cell_scores[i] > cutoff] 
        else:
            raise Exception("proportion specified not valid")
    
    # return keep indices (determined as indices not in drop indices)
    keep_idxs = [i for i in range(X.shape[0]) if i not in drop_idxs]
    
    return (keep_idxs)