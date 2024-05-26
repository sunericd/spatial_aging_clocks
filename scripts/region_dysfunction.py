'''
Runs brain subregion dysfunction analysis by comparing average absolute log fold change in gene expression for different cell types

Associated manuscript figures/tables:
- Extended Data Figures: Extended Data Fig. 3c

Inputs required:
- data/integrated_aging_coronal_celltyped_regioned_raw.h5ad - AnnData object (coronal sections dataset)

Conda environment used: `merfish.yaml`
'''

import sys
import random
import collections
import os.path
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scanpy.pp

def get_mouse_type_region_num_cells(adata, region_field_name='region'):
    '''
    Computes the number of cells in each (region, cell type, mouse) combination.
    Input:
        adata [AnnData]
            - anndata object with the transcriptional profiles.
    Returns:
        a nested dictionary, where region -> cell type -> mouse points to the number of (region, cell type, mouse) cells.
    '''
    all_mice = list(set(adata.obs['mouse_id']))
    all_ctypes = list(set(adata.obs['celltype']))
    all_regions = list(set(adata.obs[region_field_name]))
    all_genes = adata.var.index.to_list()
    all_regions_num_cells = {}

    for cur_region in all_regions:
        print(cur_region)
        cur_region_num_cells = {}
        for cur_ctype in all_ctypes:
            print(cur_ctype)
            cur_ctype_num_cells = {}
            for i, cur_mouse in enumerate(all_mice):
                is_cell_chosen = (adata.obs['celltype'] == cur_ctype) & (adata.obs[region_field_name] == cur_region) & (adata.obs['mouse_id'] == cur_mouse)
                num_chosen_cells = is_cell_chosen.sum()
                cur_ctype_num_cells[cur_mouse] = num_chosen_cells
            cur_region_num_cells[cur_ctype] = cur_ctype_num_cells
        all_regions_num_cells[cur_region] = cur_region_num_cells
    return all_regions_num_cells


def get_mouse_type_region_exp(adata, cur_region, cur_ctype, cur_mice, num_cells_per_mouse, seed=42, region_field_name='region'):
    '''
    Computes the transcriptional profile of a specific cell type, in a specific region, for a set of mice.
    The cells each of mouse are downsampled to a specified number.
    Inputs:
        adata [AnnData]
            - anndata object with the transcriptional profiles.
        cur_region [str]
            - the region for which the transcriptional profile is calculated.
        cur_ctype [str]
            - the cell type for which the transcriptional profile is calculated.
        cur_mice [str]
            - the mice for which the transcriptional profile is calculated.
        num_cells_per_mouse [int] 
            - the number of cells to sample from each mouse.
        seed [int]
            - random seed for cell sampling randomization.
        region_field_name [str]
            - the name of the field specifying the region.

    Returns: a mouse x gene pd.DataFrame with the expression of each gene for each mouse in cur_mice.
    '''
    all_genes = adata.var.index.to_list()
    returned_exp = np.empty((len(cur_mice), adata.X.shape[1]))

    assert num_cells_per_mouse > 0
    cur_exp = np.empty((len(cur_mice), adata.X.shape[1]))

    random.seed(seed)
    all_cells = list(adata.obs.index)
    for i, cur_mouse in enumerate(cur_mice):
        possible_cells = list(adata.obs.index[(adata.obs['celltype'] == cur_ctype) & (adata.obs[region_field_name] == cur_region) & (adata.obs['mouse_id'] == cur_mouse)])
        selected_cells = random.sample(possible_cells, num_cells_per_mouse)
        is_cell_chosen = adata.obs.index.isin(selected_cells)
        cur_exp = adata.X[is_cell_chosen,].mean(axis=0)
        cur_exp_norm = 250 * cur_exp / cur_exp.sum() 
        cur_exp_log = scanpy.pp.log1p(cur_exp_norm).transpose()
        returned_exp[i,] = cur_exp_log
    returned_exp_df = pd.DataFrame(returned_exp, index=cur_mice, columns=all_genes)
    return returned_exp_df


def region_dysfunction(data_path, fig_dir, drop_genes=True):
    '''
    Calculates the transcriptional changes with aging for different cell types in different regions of the mouse brain.
    Inputs:
        adata [AnnData]
            - anndata object with the transcriptional profiles.
        cur_region [str]
            - the region for which the transcriptional profile is calculated.
    '''
    adata = ad.read_h5ad(data_path)
    
    # drop genes
    if drop_genes is True:
        exclude_markers = ['Gfap', 'Crym', 'Drd2', 'Nr4a2', 'Ighm', 'Slc17a7', 'Aldoc', 'Adora2a', 'Cd4', 'C1ql3', 'Stmn2', 'Pvalb', 'Thbs4', 'Gja1', 'Atp1a2', 'C4b', 'Drd1', 'Lamp5', 'Slc1a2', 'Sparc', 'Map1lc3a', 'Tox', 'Penk', 'Gad2', 'Chat', 'Apoe', 'Aqp4', 'Sulf2', 'Sox9', 'Clu', 'Tubb3', 'Slc32a1', 'Aldh1l1', 'Spock2', 'Nfic', 'Olig1', 'Flt1', 'Pbx3', 'Pdgfra', 'Adamts3', 'Tac1', 'Cdh2', 'Slc1a3', 'Agpat3', 'Fgfr3', 'Msmo1', 'Ntm', 'Efnb2', 'Apod', 'Cd47', 'Gad1', 'Cdk5r1', 'Cfl1', 'Jak1', 'Sst', 'Sox2', 'Dpp6', 'Stub1', 'Igf2', 'Elovl5', 'Fads2', 'Trim2', 'Syt11', 'C1qa', 'Npy', 'Htt', 'Pcsk1n', 'Akt1', 'Csf1r', 'Igf1r', 'Sox11', 'Slc17a6', 'Mtor', 'C1qb', 'Sod2', 'Btg2', 'Gpm6b', 'Vcam1', 'Nr2e1', 'Parp1']
        adata = adata[:,[gene for gene in adata.var_names if gene not in exclude_markers]]
        print(adata.shape)
    
    adata.obs['mouse_id'] = ['mouse' + x for x in adata.obs['mouse_id']]
    all_regions_num_cells = get_mouse_type_region_num_cells(adata)

    all_mice_set = set(adata.obs['mouse_id'])
    bad_mice_set = set(['mouse67', 'mouse89'])
    used_mice = list(all_mice_set - bad_mice_set)

    mouse_to_age = {}
    for cur_mouse in used_mice:
        is_cur_mouse = adata.obs['mouse_id'] == cur_mouse
        cur_mouse_cell_ages = adata.obs.loc[is_cur_mouse, 'age']
        cur_mouse_age = cur_mouse_cell_ages[0]
        assert (cur_mouse_cell_ages == cur_mouse_age).all()
        mouse_to_age[cur_mouse] = cur_mouse_age

    young_mice = []
    old_mice = []
    for cur_mouse, cur_mouse_age in mouse_to_age.items():
        if cur_mouse_age < 7:
            young_mice.append(cur_mouse)
        if cur_mouse_age > 27:
            old_mice.append(cur_mouse)

    # assert that the young and old mice have the same batch distribution
    mouse_to_batch = {}
    for cur_mouse in used_mice:
        is_cur_mouse = adata.obs['mouse_id'] == cur_mouse
        cur_mouse_cell_batches = adata.obs.loc[is_cur_mouse, 'batch']
        cur_mouse_batch = cur_mouse_cell_batches[0]
        assert (cur_mouse_cell_batches == cur_mouse_batch).all()
        mouse_to_batch[cur_mouse] = cur_mouse_batch
    young_mice_batches = collections.Counter([mouse_to_batch[cur_mouse] for cur_mouse in young_mice])
    old_mice_batches = collections.Counter([mouse_to_batch[cur_mouse] for cur_mouse in old_mice])
    assert young_mice_batches == old_mice_batches


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
                        'Unknown': '0.9'}
    all_ctypes_unordered = list(set(adata.obs['celltype']))
    all_regions = list(set(adata.obs['region']))
    assert all([cur_ctype in celltype_palette for cur_ctype in all_ctypes_unordered])

    cell_types_ordered = ['Neuron-Inhibitory', 'OPC', 'Endothelial', 'Microglia', 'Oligodendrocyte', 'Astrocyte', 'Pericyte']
    all_ctypes = cell_types_ordered + list(set(all_ctypes_unordered) - set(cell_types_ordered))

    # calculate changes with age, downsampling cells so that each cell type
    # will have the same number of cells in all regions.
    # the ventricles are removed due to the low number of cells.
    num_repeats = 20
    regions_without_ven = all_regions.copy()
    regions_without_ven.remove('VEN')
    used_mice_for_age = young_mice + old_mice
    ctypes_with_enough_mice = []
    df_for_plot = pd.DataFrame(columns=['abs_diff', 'iteration', 'region', 'ctype'])
    for cur_ctype in all_ctypes:
        cur_ctype_cells_per_region = []
        for cur_region in regions_without_ven:
            num_cells_per_mouse = min([all_regions_num_cells[cur_region][cur_ctype][cur_mouse] for cur_mouse in used_mice_for_age])
            cur_ctype_cells_per_region.append(num_cells_per_mouse)
        min_num_cells = min(cur_ctype_cells_per_region)
        print(cur_ctype)
        print(min_num_cells)
        is_ctype_used = min_num_cells >= 20
        if not is_ctype_used:
            continue
        ctypes_with_enough_mice.append(cur_ctype)
        for cur_region in regions_without_ven:
            for i in range(num_repeats):
                cur_exp_mat = get_mouse_type_region_exp(adata, cur_region, cur_ctype, used_mice_for_age, min_num_cells, seed=42 + i)
                young_mice_mat = cur_exp_mat.loc[young_mice,:]
                old_mice_mat = cur_exp_mat.loc[old_mice,:]
                old_mean = old_mice_mat.mean(axis=0)
                young_mean = young_mice_mat.mean(axis=0)
                gene_diffs = old_mean - young_mean
                gene_diffs_abs = gene_diffs.abs()
                df_for_plot.loc[df_for_plot.shape[0]] = [gene_diffs_abs.mean(), i, cur_region, cur_ctype]

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    catplot_palette = [celltype_palette[cur_ctype] for cur_ctype in ctypes_with_enough_mice]
    regions_ordered = ['CC/ACO', 'CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB'] # removed VEN since low # of cells
    
    # save df for plotting
    df_for_plot.to_csv("results/dgea/genes_de_by_region_plotdf.csv")
    


if __name__ == '__main__':
    region_dysfunction(sys.argv[1], sys.argv[2])


# JUST PLOT
df_for_plot = pd.read_csv("results/dgea/genes_de_by_region_plotdf.csv")
regions_ordered = ['CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB', 'CC/ACO'] # removed VEN since low # of cells
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
                        'Unknown': '0.9'}

ctype_order = [cur_ctype for cur_ctype in celltype_palette.keys() if cur_ctype in np.unique(df_for_plot["ctype"])]
df_for_plot["ctype"] = df_for_plot["ctype"].astype('category').cat.reorder_categories(ctype_order)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
sns.catplot(data=df_for_plot, x='region', y='abs_diff', hue='ctype', kind='bar', palette=celltype_palette, order=regions_ordered,
            height=4, aspect=7/4)
plt.xlabel("Subregion", fontsize=18)
plt.ylabel("Mean Log2 Fold Change in Expression", fontsize=18)
plt.xticks(fontsize=16, rotation=90)
plt.yticks(fontsize=16)
plt.savefig('plots/dgea/genes_de_by_region_and_ctype.pdf')