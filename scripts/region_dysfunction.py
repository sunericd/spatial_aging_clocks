'''
Runs brain subregion dysfunction analysis by comparing average absolute log fold change in gene expression for different cell types

Associated manuscript figures/tables:
- Extended Data Figures: Extended Data Fig. 3c

Inputs required:
- data/integrated_aging_coronal_celltyped_regioned_raw.h5ad - AnnData object (coronal sections dataset)

Conda environment used: `requirements/merfish.txt`
'''


import pickle
import random
import os.path
import numpy as np
import pandas as pd
import anndata as ad
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import scanpy.pp

def get_mouse_type_region_num_cells(adata, region_field_name='region'):
    all_mouses = list(set(adata.obs['mouse_id']))
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
            for i, cur_mouse in enumerate(all_mouses):
                is_cell_chosen = (adata.obs['celltype'] == cur_ctype) & (adata.obs[region_field_name] == cur_region) & (adata.obs['mouse_id'] == cur_mouse)
                num_chosen_cells = is_cell_chosen.sum()
                cur_ctype_num_cells[cur_mouse] = num_chosen_cells
            cur_region_num_cells[cur_ctype] = cur_ctype_num_cells
        all_regions_num_cells[cur_region] = cur_region_num_cells
    return all_regions_num_cells


def get_mouse_type_region_exp(adata, cur_region, cur_ctype, cur_mouses, num_cells_per_mouse, seed=42, region_field_name='region'):
    all_genes = adata.var.index.to_list()
    returned_exp = np.empty((len(cur_mouses), adata.X.shape[1]))

    assert num_cells_per_mouse > 0
    cur_exp = np.empty((len(cur_mouses), adata.X.shape[1]))

    random.seed(seed)
    all_cells = list(adata.obs.index)
    for i, cur_mouse in enumerate(cur_mouses):
        possible_cells = list(adata.obs.index[(adata.obs['celltype'] == cur_ctype) & (adata.obs[region_field_name] == cur_region) & (adata.obs['mouse_id'] == cur_mouse)])
        selected_cells = random.sample(possible_cells, num_cells_per_mouse)
        is_cell_chosen = adata.obs.index.isin(selected_cells)
        cur_exp = adata.X[is_cell_chosen,].mean(axis=0)
        cur_exp_norm = 250 * cur_exp / cur_exp.sum() 
        cur_exp_log = scanpy.pp.log1p(cur_exp_norm).transpose()
        returned_exp[i,] = cur_exp_log
    returned_exp_df = pd.DataFrame(returned_exp, index=cur_mouses, columns=all_genes)
    return returned_exp_df


def region_dysfunction():
    data_path = 'data/integrated_aging_coronal_celltyped_regioned_raw.h5ad'
    fig_dir = 'plots/dgea'
    adata = ad.read_h5ad(data_path)
    adata.obs['mouse_id'] = ['mouse' + x for x in adata.obs['mouse_id']]
    #all_regions_num_cells, all_regions_exp_dicts = build_mouse_type_region_exp(adata)
    all_regions_num_cells = get_mouse_type_region_num_cells(adata)

    coarse_region_mapping = {
        'CC/ACO':'CC/ACO',
        'CTX_L1/MEN':'CTX',
        'CTX_L2/3':'CTX',
        'CTX_L4/5/6':'CTX',
        'STR_CP/ACB':'STR',
        'STR_LS/NDB':'STR',
        'VEN':'VEN',
    }
    adata.obs['region_coarse'] = [coarse_region_mapping[x] for x in adata.obs.region]
    all_coarse_regions = list(set(adata.obs['region_coarse']))
    all_coarse_regions_num_cells = get_mouse_type_region_num_cells(adata, 'region_coarse')

    all_mouses_set = set(adata.obs['mouse_id'])
    bad_mouses_set = set(['mouse67', 'mouse89'])
    used_mouses = list(all_mouses_set - bad_mouses_set)

    mouse_to_age = {}
    for cur_mouse in used_mouses:
        is_cur_mouse = adata.obs['mouse_id'] == cur_mouse
        cur_mouse_cell_ages = adata.obs.loc[is_cur_mouse, 'age']
        cur_mouse_age = cur_mouse_cell_ages[0]
        assert (cur_mouse_cell_ages == cur_mouse_age).all()
        mouse_to_age[cur_mouse] = cur_mouse_age

    # 2 mice are selected among both the young and the old, so no need to consider the batch
    # as it will cancel out
    mouse_to_batch = {}
    for cur_mouse in used_mouses:
        is_cur_mouse = adata.obs['mouse_id'] == cur_mouse
        cur_mouse_cell_batches = adata.obs.loc[is_cur_mouse, 'batch']
        cur_mouse_batch = cur_mouse_cell_batches[0]
        assert (cur_mouse_cell_batches == cur_mouse_batch).all()
        mouse_to_batch[cur_mouse] = cur_mouse_batch
    
    # This gets the youngest 5 and oldest 5 mice
    young_mouses = []
    old_mouses = []
    for cur_mouse, cur_mouse_age in mouse_to_age.items():
        if cur_mouse_age < 7:
            young_mouses.append(cur_mouse)
        if cur_mouse_age > 27:
            old_mouses.append(cur_mouse)

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
    all_ctypes = list(set(adata.obs['celltype']))
    all_regions = list(set(adata.obs['region']))
    assert all([cur_ctype in celltype_palette for cur_ctype in all_ctypes])

    num_repeats = 20 # number of repeat samplings to compute confidence interval in plot
    regions_without_ven = all_regions.copy()
    regions_without_ven.remove('VEN')
    used_mouses_for_age = young_mouses + old_mouses
    ctypes_with_enough_mouses = []
    df_for_plot = pd.DataFrame(columns=['abs_diff', 'iteration', 'region', 'ctype'])
    for cur_ctype in all_ctypes:
        cur_ctype_cells_per_region = []
        for cur_region in regions_without_ven:
            num_cells_per_mouse = min([all_regions_num_cells[cur_region][cur_ctype][cur_mouse] for cur_mouse in used_mouses_for_age])
            cur_ctype_cells_per_region.append(num_cells_per_mouse)
        min_num_cells = min(cur_ctype_cells_per_region)
        print(cur_ctype)
        print(min_num_cells)
        is_ctype_used = min_num_cells >= 20 # minimum number of cells needed per sample to show
        if not is_ctype_used:
            continue
        ctypes_with_enough_mouses.append(cur_ctype)
        for cur_region in regions_without_ven:
            for i in range(num_repeats):
                cur_exp_mat = get_mouse_type_region_exp(adata, cur_region, cur_ctype, used_mouses_for_age, min_num_cells, seed=42 + i)
                young_mouses_mat = cur_exp_mat.loc[young_mouses,:]
                old_mouses_mat = cur_exp_mat.loc[old_mouses,:]
                old_mean = old_mouses_mat.mean(axis=0)
                young_mean = young_mouses_mat.mean(axis=0)
                gene_diffs = old_mean - young_mean
                gene_diffs_abs = gene_diffs.abs()
                df_for_plot.loc[df_for_plot.shape[0]] = [gene_diffs_abs.mean(), i, cur_region, cur_ctype]


    # and now again, with corase regions:
    coarse_regions_without_ven = all_coarse_regions.copy()
    coarse_regions_without_ven.remove('VEN')
    coarse_ctypes_with_enough_mouses = []
    coarse_df_for_plot = pd.DataFrame(columns=['abs_diff', 'iteration', 'region', 'ctype'])
    for cur_ctype in all_ctypes:
        cur_ctype_cells_per_region = []
        for cur_region in coarse_regions_without_ven:
            num_cells_per_mouse = min([all_coarse_regions_num_cells[cur_region][cur_ctype][cur_mouse] for cur_mouse in used_mouses_for_age])
            cur_ctype_cells_per_region.append(num_cells_per_mouse)
        min_num_cells = min(cur_ctype_cells_per_region)
        print(cur_ctype)
        print(min_num_cells)
        is_ctype_used = min_num_cells >= 20
        if not is_ctype_used:
            continue
        coarse_ctypes_with_enough_mouses.append(cur_ctype)
        for cur_region in coarse_regions_without_ven:
            for i in range(num_repeats):
                cur_exp_mat = get_mouse_type_region_exp(adata, cur_region, cur_ctype, used_mouses_for_age, min_num_cells, seed=42 + i, region_field_name='region_coarse')
                young_mouses_mat = cur_exp_mat.loc[young_mouses,:]
                old_mouses_mat = cur_exp_mat.loc[old_mouses,:]
                old_mean = old_mouses_mat.mean(axis=0)
                young_mean = young_mouses_mat.mean(axis=0)
                gene_diffs = old_mean - young_mean
                gene_diffs_abs = gene_diffs.abs()
                coarse_df_for_plot.loc[coarse_df_for_plot.shape[0]] = [gene_diffs_abs.mean(), i, cur_region, cur_ctype]

    catplot_palette = [celltype_palette[cur_ctype] for cur_ctype in ctypes_with_enough_mouses]
    coarse_catplot_palette = [celltype_palette[cur_ctype] for cur_ctype in coarse_ctypes_with_enough_mouses]

    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42

    regions_ordered = ['CC/ACO', 'CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB'] # removed VEN since low # of cells
    
    # save df for plotting
    df_for_plot.to_csv("results/dgea/genes_de_by_region_plotdf.csv")
    
    ctype_order = [cur_ctype for cur_ctype in celltype_palette.keys() if cur_ctype in ctypes_with_enough_mouses]
    df_for_plot["ctype"] = df_for_plot["ctype"].astype('category').cat.reorder_categories(ctype_order)
        
    fig, ax = plt.subplots(figsize=(5, 7))
    catplot_ret = sns.catplot(data=df_for_plot, x='abs_diff', y='region', hue='ctype', kind='bar', palette=catplot_palette, order=regions_ordered, ax=ax)
    plt.ylabel("Subregion", fontsize=18)
    plt.xlabel("Mean Log2 Fold Change in Expression", fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    catplot_ret.savefig(os.path.join(fig_dir, 'genes_de_by_region_and_ctype2.pdf'))

    
# RUN
region_dysfunction()

# JUST PLOT
df_for_plot = pd.read_csv("results/dgea/genes_de_by_region_plotdf.csv")
regions_ordered = ['CC/ACO', 'CTX_L1/MEN', 'CTX_L2/3', 'CTX_L4/5/6', 'STR_CP/ACB', 'STR_LS/NDB'] # removed VEN since low # of cells
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
sns.catplot(data=df_for_plot, x='abs_diff', y='region', hue='ctype', kind='bar', palette=celltype_palette, order=regions_ordered,
            height=8, aspect=5/8)
plt.ylabel("Subregion", fontsize=18)
plt.xlabel("Mean Log2 Fold Change in Expression", fontsize=18)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.savefig('plots/dgea/genes_de_by_region_and_ctype2.pdf')
