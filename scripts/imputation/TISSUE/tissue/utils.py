# Contains utility functions for TISSUE

import numpy as np
import pandas as pd
import anndata as ad
import os


def large_save(adata, dirpath):
    '''
    Saves anndata objects by saving each obsm value with its {key}.csv as pandas dataframe
    Saves each uns value that is a dataframe with uns/{key}.csv as pandas dataframe
    Then saves the anndata object with obsm removed.
    
    Parameters
    ----------
        adata [AnnData] - AnnData object to save
        
        dirpath [str] - path to directory for where to save the h5ad and csv files; will create if not existing
            adata will be saved as {dirpath}/adata.h5ad
            obsm will be saved as {dirpath}/{key}.csv
        
    Returns
    -------
        Saves anndata object in "large" folder format
    '''
    # check if dirpath exists; else create it
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)
    
    # extract the obsm metadata and save it as separate csv files
    for key, value in adata.obsm.items():
        df = pd.DataFrame(value)
        df.to_csv(os.path.join(dirpath, f"{key}.csv"), index=False)

    # remove the obsm metadata from the anndata object
    adatac = adata.copy()
    adatac.obsm = {}
    
    # extract the uns metadata and save it as separate csv files
    del_keys = []
    for key, value in adatac.uns.items():
        if isinstance(value, pd.DataFrame):
            if not os.path.exists(os.path.join(dirpath,"uns")):
                os.makedirs(os.path.join(dirpath,"uns"))
            df = pd.DataFrame(value)
            df.to_csv(os.path.join(dirpath,"uns",f"{key}.csv"), index=False)
            del_keys.append(key)
    
    # remove uns metadata from the anndata object
    for key in del_keys:
        del adatac.uns[key]

    # save the new anndata object
    adatac.write(os.path.join(dirpath, "adata.h5ad"))



def large_load(dirpath, skipfiles=[]):
    '''
    Loads in anndata and associated pandas dataframe csv files to be added to obsm metadata and uns metadata.
    Input is the directory path to the output directory of large_save()
    
    Parameters
    ----------
        dirpath [str] - path to directory for where outputs of large_save() are located
        skipfiles [list] - list of filenames to exclude from anndata object
    
    Returns
    -------
        adata - AnnData object loaded from dirpath along with all obsm and uns key values added to metadata
    '''
    # read h5ad anndata object
    adata = ad.read_h5ad(os.path.join(dirpath, "adata.h5ad"))
    
    # read and load in obsm from CSV files
    for fn in os.listdir(dirpath):
        if (".csv" in fn) and (fn not in skipfiles):
            df = pd.read_csv(os.path.join(dirpath, fn))
            df.index = adata.obs_names
            key = fn.split(".")[0]
            adata.obsm[key] = df
            
    # read and load any usn metadata from CSV files
    if os.path.isdir(os.path.join(dirpath,"uns")):
        for fn in os.listdir(os.path.join(dirpath,"uns")):
            if (".csv" in fn) and (fn not in skipfiles):
                df = pd.read_csv(os.path.join(dirpath,"uns",fn))
                key = fn.split(".")[0]
                adata.uns[key] = df
            
    return(adata)


def convert_adata_to_dataupload (adata, savedir):
    '''
    Saves AnnData object into TISSUE input directory
    
    Parameters
    ----------
        adata - AnnData object to be saved with all metadata in adata.obs and spatial coordinates in adata.obsm['spatial']
        savedir [str] - path to existing directory to save the files for TISSUE loading
        
    Returns
    -------
        Saves all TISSUE input files into the specified directory for the given AnnData object
        
    NOTE: You will need to independently include scRNA_count.txt in savedir for TISSUE inputs to be complete
    '''
    locations = pd.DataFrame(adata.obsm['spatial'], columns=['x','y'])
    locations.to_csv(os.path.join(savedir,"Locations.txt"), sep="\t", index=False)
    
    df = pd.DataFrame(adata.X, columns=adata.var_names)
    df.to_csv(os.path.join(savedir,"Spatial_count.txt"), sep="\t", index=False)
    
    meta = pd.DataFrame(adata.obs)
    meta.to_csv(os.path.join(savedir,"Metadata.txt"))