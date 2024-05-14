'''
Classes for GNN model and dataset object using PyG
'''


# import key packages and libraries
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import os
from sklearn.neighbors import BallTree

from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal

import random

from ageaccel_proximity import *

import networkx as nx

import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.utils import k_hop_subgraph, one_hot, to_networkx
from torch_geometric.utils.convert import from_scipy_sparse_matrix
from torch.nn import Linear, Sequential, BatchNorm1d, ReLU, ModuleList
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GINConv, SAGEConv, global_mean_pool, global_add_pool, global_max_pool
from torch.nn.modules.loss import _Loss


# Dataset Class

class SpatialAgingCellDataset(Dataset):
    '''
    Class for building spatial cell subgraphs from the MERFISH anndata file
        - Nodes are cells and featurized by one-hot encoding of cell type
        - Edges are trimmed Delaunay spatial graph connections
        - Graphs are k-hop neighborhoods around cells and labeled by average peripheral age acceleration
    
    Relies on get_age_acceleration() and build_spatial_graph() from ageaccel_proximity.py
        Use `from ageaccel_proximity import *` when importing libraries
        
    Arguments:
        root [str] - root directory path
        transform [None] - not implemented
        pre_transform [None] - not implemented
        raw_filepath [str] - path to anndata .h5ad file with MERFISH data
        processed_folder_name [str] - path to save processed data files
        subfolder_name [str] - name of subfolder to save in (e.g. "train")
        target [str] - name of target label to use ("aging", "age", "num_neuron")
        node_feature [str] - how to featurize nodes ("celltype", "celltype_region")
        sub_id [str] - key in adata.obs to separate graphs by id
        use_ids [lst of str] - list of sub_id values to use
        center_celltypes [str or lst] - 'all' to use all cell types, otherwise list of cell types to draw subgraphs from
        num_cells_per_ct_id [str] - number of cells per cell type per id to take
        k_hop [int] - k-hop neighborhood subgraphs to take
        augment_hop [int] - k-hop neighbors to also take induced subgraphs from (augments number of graphs)
        augment_cutoff ['auto', 0 <= float < 1] - quantile cutoff in absolute value of label to perform augmentation (to balance labels)
        dispersion_factor [0 <= float < 1] - factor for dispersion of augmentation sampling of rare graph labels; higher means more rare samples
        radius_cutoff [float] - radius cutoff for Delaunay triangulation edges
        restricted_celltype_subset [lst of str] - list of cell types to use for neighborhood age acceleration calculation
        celltypes_to_index [dict] - dictionary mapping cell type labels to integer index
        regions_to_index [dict] - dictionary mapping region labels to integer index
    '''
    def __init__(self, 
                 root=".", 
                 transform=None, 
                 pre_transform=None, 
                 raw_filepath="results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20.h5ad",
                 processed_folder_name="data/gnn_datasets",
                 subfolder_name=None,
                 target="aging",
                 node_feature="celltype",
                 sub_id="mouse_id",
                 use_ids=[],
                 center_celltypes='all',
                 num_cells_per_ct_id=1,
                 k_hop=2,
                 augment_hop=0,
                 augment_cutoff=0,
                 dispersion_factor=0,
                 radius_cutoff=200,
                 restricted_celltype_subset=['Neuron-Excitatory','Neuron-MSN','Astrocyte','Microglia','Oligodendrocyte','OPC','Endothelial','Pericyte','VSMC','Ependymal','Neuroblast','NSC','Macrophage', 'T cell'],
                 celltypes_to_index = {
                                     'Neuron-Excitatory' : 0,
                                     'Neuron-Inhibitory' : 1,
                                     'Neuron-MSN' : 2, 
                                     'Astrocyte' : 3, 
                                     'Microglia' : 4, 
                                     'Oligodendrocyte' : 5, 
                                     'OPC' : 6,
                                     'Endothelial' : 7, 
                                     'Pericyte' : 8, 
                                     'VSMC' : 9, 
                                     'VLMC' : 10,
                                     'Ependymal' : 11, 
                                     'Neuroblast' : 12, 
                                     'NSC' : 13,  
                                     'Macrophage' : 14, 
                                     'Neutrophil' : 15,
                                     'T cell' : 16, 
                                     'B cell' : 17,
                                    },
                 regions_to_index = {
                                     'CC/ACO': 0, 
                                     'CTX_L1/MEN': 1, 
                                     'CTX_L2/3': 1, 
                                     'CTX_L4/5/6': 1, 
                                     'STR_CP/ACB': 2,
                                     'STR_LS/NDB': 2, 
                                     'VEN': 3,
                                     },
                ):
    
        self.root=root
        self.transform=transform
        self.pre_transform=pre_transform
        self.raw_filepath=raw_filepath
        self.processed_folder_name=processed_folder_name
        self.subfolder_name=subfolder_name
        self.target=target
        self.node_feature=node_feature
        self.sub_id=sub_id
        self.use_ids=use_ids
        self.center_celltypes=center_celltypes
        self.num_cells_per_ct_id=num_cells_per_ct_id
        self.k_hop=k_hop
        self.augment_hop=augment_hop
        self.augment_cutoff=augment_cutoff
        self.dispersion_factor=dispersion_factor
        self.radius_cutoff=radius_cutoff
        self.restricted_celltype_subset=restricted_celltype_subset
        self.celltypes_to_index=celltypes_to_index
        self.regions_to_index=regions_to_index
        self._indices = None

    def indices(self):
        return range(self.len()) if self._indices is None else self._indices
    
    @property
    def processed_dir(self) -> str:
        if self.augment_cutoff == 'auto':
            aug_key = self.augment_cutoff
        else:
            aug_key = int(self.augment_cutoff*100)
        celltype_firstletters = "".join([x[0] for x in self.center_celltypes])
        data_dir = f"{self.target}_{self.num_cells_per_ct_id}per_{self.k_hop}hop_{self.augment_hop}C{aug_key}aug_{self.radius_cutoff}delaunay_{self.node_feature}Feat_{celltype_firstletters}"
        if self.subfolder_name is not None:
            return os.path.join(self.root, self.processed_folder_name, data_dir, self.subfolder_name)
        else:
            return os.path.join(self.root, self.processed_folder_name, data_dir)

    @property
    def processed_file_names(self):
        return sorted([f for f in os.listdir(self.processed_dir) if f.endswith('.pt')])
    
    def process(self):
        
        # Create / overwrite directory
        if self.augment_cutoff == 'auto':
            aug_key = self.augment_cutoff
        else:
            aug_key = int(self.augment_cutoff*100)
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)
        else:
            print ("Dataset already exists!")
            return()
        
        # load raw data
        adata = sc.read_h5ad(self.raw_filepath)
        
        # compute age acceleration
        get_age_acceleration (adata)
        
        # make and save subgraphs
        subgraph_count = 0
        
        sub_ids_arr = np.intersect1d(np.unique(adata.obs[self.sub_id]), np.array(self.use_ids))
        
        for sid in sub_ids_arr:

            # subset to each sample
            sub_adata = adata[(adata.obs[self.sub_id]==sid)]

            # Delaunay triangulation with pruning of > 200um distances
            build_spatial_graph(sub_adata, method="delaunay")
            sub_adata.obsp['spatial_connectivities'][sub_adata.obsp['spatial_distances']>self.radius_cutoff] = 0
            sub_adata.obsp['spatial_distances'][sub_adata.obsp['spatial_distances']>self.radius_cutoff] = 0

            # convert graphs to PyG format
            edge_index, edge_att = from_scipy_sparse_matrix(sub_adata.obsp['spatial_connectivities'])
            
            ### Construct Node Labels
            if self.node_feature not in ["celltype", "celltype_region"]:
                raise Exception (f"'node_feature' value of {self.node_feature} not recognized")
            
            if "celltype" in self.node_feature:
                # get cell type one hot encoding
                node_labels = torch.tensor([self.celltypes_to_index[x] for x in sub_adata.obs["celltype"]])
                node_labels = one_hot(node_labels)
            
            if "region" in self.node_feature:
                # get region
                reg_labels = torch.tensor([self.regions_to_index[x] for x in sub_adata.obs["region"]])
                reg_labels = one_hot(reg_labels)
                node_labels = torch.cat((node_labels, reg_labels), 1).float()
                        
            ### Get Indices of Random Center Cells
            cell_idxs = []
            
            if self.center_celltypes == "all":
                center_celltypes_to_use = np.unique(sub_adata.obs["celltype"])
            else:
                center_celltypes_to_use = self.center_celltypes
            
            for ct in center_celltypes_to_use:
                np.random.seed(444)
                idxs = np.random.choice(np.arange(sub_adata.shape[0])[sub_adata.obs["celltype"]==ct],
                                        np.min([self.num_cells_per_ct_id, np.sum(sub_adata.obs["celltype"]==ct)]),
                                        replace=False)
                cell_idxs = np.concatenate((cell_idxs, idxs))

            ### Extract K-hop Subgraphs
            
            graph_labels = [] # for computing quantiles later
            for cidx in cell_idxs:
                # get subgraph
                sub_node_labels, sub_edge_index, graph_label = self.subgraph_from_index(int(cidx), edge_index, node_labels, sub_adata)
                
                # filter out tiny subgraphs
                if len(sub_node_labels) > 2*self.k_hop:
                    
                    # append graph_label (for computing augmentation quantiles)
                    graph_labels.append(graph_label)
                    
                    # make PyG Data object
                    subgraph_data = Data(x = sub_node_labels,
                                         edge_index = sub_edge_index,
                                         y = torch.tensor([graph_label]))

                    # save object
                    torch.save(subgraph_data,
                               os.path.join(self.processed_dir,f"g{subgraph_count}.pt"))
                    subgraph_count += 1
                    
            ### Selective Graph Augmentation
            
            # get augmentation indices
            if self.augment_hop > 0:
                augment_idxs = []
                for cidx in cell_idxs:
                    # get subgraph and get node indices of all nodes
                    sub_nodes, sub_edge_index, center_node_idx, edge_mask = k_hop_subgraph(
                                                                            int(cidx),
                                                                            self.augment_hop, 
                                                                            edge_index,
                                                                            relabel_nodes=True)
                    augment_idxs = np.concatenate((augment_idxs,sub_nodes.detach().numpy()))
                
                augment_idxs = np.unique(augment_idxs) # remove redundancies
                
                avg_aug_size = len(augment_idxs)/len(cell_idxs) # get average number of augmentations per center cell
            
                # compute augmentation cutoff
                if self.augment_cutoff == "auto":
                    bins, bin_edges = np.histogram(graph_labels, bins=5)
                    bins = np.concatenate((bins[0:1], bins, bins[-1:])) # expand edge bins with duplicate counts
                else:
                    absglcutoff = np.quantile(np.abs(graph_labels), self.augment_cutoff)

                # get subgraphs and save for augmentation
                for cidx in augment_idxs:
                                        
                    # get subgraph
                    sub_node_labels, sub_edge_index, graph_label = self.subgraph_from_index(int(cidx), edge_index, node_labels, sub_adata)
                                        
                    # augmentation selection conditions
                    if self.augment_cutoff == "auto": # probabilistic
                        curr_bin = bins[np.digitize(graph_label,bin_edges)] # get freq of current bin
                        prob_aug = (np.max(bins) - curr_bin) / (curr_bin * avg_aug_size * (1-self.dispersion_factor))
                        do_aug = (random.random() < prob_aug) # augment with probability based on max bin size
                    else: # by quantile cutoff
                        do_aug = (np.abs(graph_label) >= absglcutoff) # if pass graph label cutoff then augment
                    
                    # save augmented graphs if conditions met
                    if (len(sub_node_labels) > 2*self.k_hop) and (do_aug):
                        
                        # make PyG Data object
                        subgraph_data = Data(x = sub_node_labels,
                                             edge_index = sub_edge_index,
                                             y = torch.tensor([graph_label]))

                        # save object
                        torch.save(subgraph_data,
                                   os.path.join(self.processed_dir,f"g{subgraph_count}.pt"))
                        subgraph_count += 1

    
    def subgraph_from_index (self, cidx, edge_index, node_labels, sub_adata):
        '''
        Method used by self.process to extract subgraph and properties based on a cell index (cidx) and edge_index and node_labels and sub_adata
        '''
        # get subgraph
        sub_nodes, sub_edge_index, center_node_idx, edge_mask = k_hop_subgraph(
                                                                int(cidx),
                                                                self.k_hop, 
                                                                edge_index,
                                                                relabel_nodes=True)
        # get node values
        sub_node_labels = node_labels[sub_nodes,:]

        # label graphs with neighborhood age acceleration (center cell removed)
        if self.target == "aging":
            without_center_idxs = np.array(sub_nodes)[sub_nodes!=cidx]
            graph_label = np.float32(np.nanmean(sub_adata.obs["normalized_age_acceleration"].values[without_center_idxs]))
        elif self.target == "age":
            without_center_idxs = np.array(sub_nodes)[sub_nodes!=cidx]
            graph_label = np.float32(np.nanmean(sub_adata.obs["age"].values[without_center_idxs]))
        elif self.target == "num_neuron":
            without_center_idxs = np.array(sub_nodes)[sub_nodes!=cidx]
            graph_label = np.float32(np.sum(sub_adata[without_center_idxs,:].obs["celltype"].isin(['Neuron-Excitatory','Neuron-Inhibitory','Neuron-MSN'])))
        else:
            raise Exception ("'target' not recognized")
        
        return (sub_node_labels, sub_edge_index, graph_label)
    

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(os.path.join(self.processed_dir, f'g{idx}.pt'))
        return data
		

# Model class

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels, input_dim,
                 num_layers=3, method="GCN", pool="add"):
        super(GNN, self).__init__()
        torch.manual_seed(444)
        
        self.method = method
        self.pool = pool
        
        if self.method == "GCN":
            self.conv1 = GCNConv(input_dim, hidden_channels)
            self.convs = ModuleList([GCNConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
        
        elif self.method == "GIN":
            self.conv1 = GINConv(
                            Sequential(
                                Linear(input_dim, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          )
            self.convs = ModuleList([GINConv(
                            Sequential(
                                Linear(hidden_channels, hidden_channels),
                                BatchNorm1d(hidden_channels),
                                ReLU(),
                                Linear(hidden_channels, hidden_channels)
                            )
                          ) for _ in range(num_layers - 1)])
        
        elif self.method == "SAGE":
            self.conv1 = SAGEConv(input_dim, hidden_channels)
            self.convs = ModuleList([SAGEConv(hidden_channels, hidden_channels) for _ in range(num_layers - 1)])
            
        else:
            raise Exception("'method' not recognized.")
        
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, batch):
                    
        # node embeddings 
        x = F.relu(self.conv1(x, edge_index))
        for layer_idx, conv in enumerate(self.convs):
            if layer_idx < len(self.convs) - 1:
                x = F.relu(conv(x, edge_index))
            else:
                x = conv(x, edge_index)

        # pooling and readout
        if self.pool == "mean":
            x = global_mean_pool(x, batch)
        elif self.pool == "add":
            x = global_add_pool(x, batch)
        elif self.pool == "max":
            x = global_max_pool(x, batch)
        else:
            raise Exception ("'pool' not recognized")

        # final prediction
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        
        return x
    
    
# balanced MSE loss function
def bmc_loss(pred, target, noise_var):
    """Compute the Balanced MSE Loss (BMC) between `pred` and the ground truth `targets`.
    Args:
      pred: A float tensor of size [batch, 1].
      target: A float tensor of size [batch, 1].
      noise_var: A float number or tensor.
    Returns:
      loss: A float tensor. Balanced MSE Loss.
    """
    logits = - (pred - target.T).pow(2) / (2 * noise_var)   # logit size: [batch, batch]
    loss = F.cross_entropy(logits, torch.arange(pred.shape[0]))     # contrastive-like loss
    loss = loss * (2 * noise_var).detach()  # optional: restore the loss scale, 'detach' when noise is learnable 

    return loss

class BMCLoss(_Loss):
    def __init__(self, init_noise_sigma):
        super(BMCLoss, self).__init__()
        self.noise_sigma = torch.nn.Parameter(torch.tensor(init_noise_sigma))

    def forward(self, pred, target):
        noise_var = self.noise_sigma ** 2
        return bmc_loss(pred, target, noise_var)



    
# negative pearson correlation loss
def npcc_loss(pred, target):
    """
    Negative pearson correlation as loss
    """
    
    # Alternative formulation
    x = torch.flatten(pred)
    y = torch.flatten(target)
    
    mean_x = torch.mean(x)
    mean_y = torch.mean(y)
    xm = x.sub(mean_x)
    ym = y.sub(mean_y)
    r_num = xm.dot(ym)
    r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
    r_val = r_num / r_den
    
    loss = 1-r_val

    return loss
    
class Neg_Pearson_Loss(_Loss):
    def __init__(self):
        super(Neg_Pearson_Loss, self).__init__()
        return

    def forward(self, pred, target):
        return npcc_loss(pred, target)






def train(model, loader, criterion, optimizer):
    model.train()

    for data in loader:  # Iterate in batches over the training dataset.
        out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.

def test(model, loader, loss, criterion):
    model.eval()

    errors = []
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        if loss == "mse":
            errors.append(F.mse_loss(out, data.y.unsqueeze(1)).sqrt().item())
        elif loss == "l1":
            errors.append(F.l1_loss(out, data.y.unsqueeze(1)).item())
        elif loss == "balanced_mse":
            errors.append(bmc_loss(out, data.y.unsqueeze(1), criterion.noise_sigma**2).item())
        elif loss == "npcc":
            errors.append(npcc_loss(out, data.y.unsqueeze(1)).item())
        
    return np.mean(errors)  # Derive ratio of correct predictions.