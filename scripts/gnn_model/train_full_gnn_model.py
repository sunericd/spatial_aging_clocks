'''
Runs training of the GNN model on the entire dataset.
Trained GNN model will be used for in silico perturbation modeling.

Inputs required:
- "results/clocks/anndata/lasso_loocv_predicted_age_correlation_n30_spatialsmoothonsmooth_alpha08_nneigh20.h5ad" - AnnData object with predicted ages on the coronal section dataset (generated from 3A_clocks_single_cell.ipynb)

Conda environment used: `requirements/merfish_gnn.txt`
'''

# import key packages and libraries
import numpy as np
import pandas as pd
import scanpy as sc
import squidpy as sq
import anndata as ad
from scipy.stats import pearsonr, spearmanr, ttest_ind
import pickle
import copy
import os
from sklearn.neighbors import BallTree

from scipy.stats import mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from decimal import Decimal

import random

from aging_gnn_model import *

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

import argparse

os.chdir("../..")


# set up arguments
parser = argparse.ArgumentParser()
parser.add_argument("k_hop", help="k-hop neighborhood size", type=int)
parser.add_argument("augment_hop", help="number of hops to take for graph augmentation", type=int)
parser.add_argument("center_celltypes", help="cell type labels to center graphs on, separated by comma", type=str)
parser.add_argument("node_feature", help="node features key, e.g. 'celltype_age_region'", type=str)
parser.add_argument("learning_rate", help="learning rate", type=float)
parser.add_argument("loss", help="loss: balanced_mse, npcc, mse, l1", type=str)
args = parser.parse_args()

# load parameters from arguments
k_hop = args.k_hop
augment_hop = args.augment_hop
center_celltypes = args.center_celltypes.split(",")
node_feature = args.node_feature
learning_rate = args.learning_rate
loss = args.loss

# init dataset with settings
train_dataset = SpatialAgingCellDataset(subfolder_name="train",
                                        target="aging",
                                        k_hop=k_hop,
                                        augment_hop=augment_hop,
                                        node_feature=node_feature,
                                        num_cells_per_ct_id=100,
                                        center_celltypes=center_celltypes,
                                use_ids=['1','101','14','19','30','38','42',
                                         '46','53','61','7','70','75','80',
                                         '86','97'])

test_dataset = SpatialAgingCellDataset(subfolder_name="test",
                                       target="aging",
                                       k_hop=k_hop,
                                       augment_hop=augment_hop,
                                       node_feature=node_feature,
                                       num_cells_per_ct_id=100,
                                       center_celltypes=center_celltypes,
                                  use_ids=["11","33",
                                           "57","93"])
                                        
# process and save data files of subgraphs -- ONLY NEED TO RUN ONCE
test_dataset.process()
print("Finished processing test dataset")
train_dataset.process()
print("Finished processing train dataset")

# concatenate datasets (train on both data combined)
all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

# data loaders
all_loader = DataLoader(all_dataset, batch_size=64, shuffle=True)

print(len(all_dataset))

# init GNN model
model = GNN(hidden_channels=16,
            input_dim=int(train_dataset.get(0).x.shape[1]),
            method="GIN", pool="add", num_layers=k_hop)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# get loss
if loss == "mse":
    criterion = torch.nn.MSELoss()
elif loss == "l1":
    criterion = torch.nn.L1Loss()
elif loss == "balanced_mse":
    criterion = BMCLoss(0.1) # init noise_sigma
    optimizer.add_param_group({'params': criterion.noise_sigma, 'lr':learning_rate, 'name': 'noise_sigma'})
elif loss == "npcc":
    criterion = Neg_Pearson_Loss()

# train model
training_results = {"metric":loss, "epoch":[], "train":[]}
best_mse = np.inf

for epoch in range(1, 50):

    train(model, all_loader, criterion, optimizer)
    train_mse = test(model, all_loader, loss, criterion)
    
    if train_mse < best_mse: # if model improved then save
        best_model = copy.deepcopy(model)
        best_mse = train_mse
    
    if loss == "mse":
        print(f'Epoch: {epoch:03d}, Train MSE: {train_mse:.4f}')
    elif loss == "l1":
        print(f'Epoch: {epoch:03d}, Train L1: {train_mse:.4f}')
    elif loss == "balanced_mse":
        print(f'Epoch: {epoch:03d}, Train BMC: {train_mse:.4f}')
    elif loss == "npcc":
        print(f'Epoch: {epoch:03d}, Train NPCC: {train_mse:.4f}')
        
    training_results["epoch"].append(epoch)
    training_results["train"].append(train_mse)    

# SAVE RESULTS
model_dirname = loss+f"_{learning_rate:.0e}".replace("-","n")
save_dir = os.path.join("results/gnn",train_dataset.processed_dir.split("/")[-2],model_dirname)
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# save best model
torch.save(best_model.state_dict(), os.path.join(save_dir, "all_best_model.pth"))
print("Saved best model")

# save final model
torch.save(model.state_dict(), os.path.join(save_dir, "all_model.pth"))
print("Saved final model")

# save results
with open(os.path.join(save_dir, "all_training.pkl"), 'wb') as f:
    pickle.dump(training_results, f)
print("Saved training logs")