# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# %%
import sys
import os

sys.path.append('..')

from data_preprocessing.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
import torch
import numpy as np
from tqdm import tqdm
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN
from branca.colormap import linear
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf



# %%
def visualize_y(y, idx_to_node_ids):
    # input: nodes, feature, timesteps
    # output: feature, timestep, nodes -> timestep, nodes
    y_t = np.transpose(y, (1, 2, 0)).squeeze()
    
    occurance_list = [[ i for i,a in enumerate(ab) if a==1 ]for ab in y_t]
    occurance_list = [j for sub in occurance_list for j in sub]
    occurance_dict = {v:0 for v in idx_to_node_ids.values()}
    
    for o in occurance_list:
        occurance_dict[idx_to_node_ids[o]] += 1
    
    regions = pd.read_pickle("../data/regions_enhanced.pkl")
    visualize_df = regions.copy()
    visualize_df["visited"] = pd.Series(occurance_dict)
    # TODO: add number visited
    
    cmap = linear.Blues_09.scale(0, max(occurance_dict.values()))
    return visualize_df.explore("visited", cmap=cmap)


# %%
x_features = ["hex_work","hex_errand", "hex_leisure","activity_work","activity_errand","activity_leisure", "visited"]
y_features = ["visited"]

# max datapoints 33799, 10000 gives memory problems 
loader = TrajectoryDatasetLoader("../data/dataset1.json", x_features, y_features) # 1000
dataset = loader.get_dataset()

# 33799
print("Dataset type:  ", dataset)
print("Number of samples / sequences: ",  dataset.snapshot_count)

# %%
# timesteps, nodes, features
loader.X[0,:,3].shape
loader.features_to_ids

# %%
fig, axes = plt.subplots(1,3, figsize=(12,4))
# timesteps, nodes, features
sns.histplot(loader.X[0,:,loader.features_to_ids["hex_work"]], ax = axes[0])
axes[0].set(xlabel='hex_work')
sns.histplot(loader.X[0,:,loader.features_to_ids["hex_leisure"]], ax = axes[1])
axes[1].set(xlabel='hex_leisure')
sns.histplot(loader.X[0,:,loader.features_to_ids["hex_errand"]], ax = axes[2])
axes[2].set(xlabel='hex_errand')
plt.show()

# %%
fig, axes = plt.subplots(1,3, figsize=(12,4))
# timesteps, nodes, features
sns.histplot(loader.X[:,0,loader.features_to_ids["activity_work"]], ax = axes[0])
axes[0].set(xlabel='activity_work')
sns.histplot(loader.X[:,0,loader.features_to_ids["activity_leisure"]], ax = axes[1])
axes[0].set(xlabel='activity_leisure')
sns.histplot(loader.X[:,0,loader.features_to_ids["activity_errand"]], ax = axes[2])
axes[0].set(xlabel='activity_errand')
plt.show()

# %%
# purpose
ax =sns.histplot(loader.X[:,0,loader.features_to_ids["purpose"]])
ax.set(xlabel='purpose')
print(loader.ids_to_purposes)


# %%
# time

# %%
# visited
to_show =  "cvisited" # "visited" #

activity_work_vec = loader.X[:,0,loader.features_to_ids["activity_work"]]
activity_leisure_vec = loader.X[:,0,loader.features_to_ids["activity_leisure"]]
activity_errand_vec = loader.X[:,0,loader.features_to_ids["activity_errand"]]
purpose_vec = loader.X[:,0,loader.features_to_ids["purpose"]]
visited_vec = loader.X[:,:,loader.features_to_ids["visited"]]
user_id_vec = loader.X[:,:,loader.features_to_ids["user_id"]]

sum_conditional_visited = np.zeros(visited_vec.shape[1])
for i,(w,l,e,p, u) in enumerate(zip(activity_work_vec, activity_leisure_vec, activity_errand_vec, purpose_vec, user_id_vec)):
    
    # CONFIGURE CONDITION HERE
    condition = (w >= 5)
    if condition:
        sum_conditional_visited += visited_vec[i,:]

gdf = pd.read_pickle("../data/regions_enhanced.pkl")
gdf["visited"] = np.sum(loader.X[:,:,loader.features_to_ids["visited"]], axis=0).astype(int)
gdf["cvisited"] = sum_conditional_visited
cmap = linear.Blues_09.scale(0, gdf[to_show].max())
gdf.explore(to_show, cmap=cmap)


# %%
example = next(iter(dataset))
visualize_y(example.y, loader.ids_to_hex_names)
