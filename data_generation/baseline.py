# %%
import sys
import pandas as pd
import pickle
import random
import torch

sys.path.append('..')
sys.path.append('../../')

from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from torch_geometric_temporal.signal import temporal_signal_split
from master_thesis.utils.utils import get_distances_df
import networkx as nx

# %%
resolution = 8

x_features = ["hex_work","hex_errand", "hex_leisure","activity_work","activity_errand","activity_leisure", "visited"]
y_features = ["visited"]
visited_id = x_features.index("visited")

dataset_name = f"dataset_res{resolution}"
tsin = 3
tsout =3

loader = TrajectoryDatasetLoader(f"../data/{dataset_name}_v2.json", tsin,tsout)
dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)

resolution = loader.resolution
ids_to_hex_names = loader.ids_to_hex_names
hex_names_to_ids = loader.hex_names_to_ids

# 33799
# print("Dataset type:  ", dataset)
# print("Number of samples / sequences: ",  dataset.snapshot_count)

# %%
ns = test_set.snapshot_count
ns

# %%
regions = pd.read_pickle(f"../data/regions_enhanced_res{resolution}_v2.pkl")
distances_df = get_distances_df(regions)

# %%
graph_name = f"base_graph_res{resolution}.pickle"
graph = pickle.load(open("../data/" +graph_name, 'rb'))
graph

# %%
from tqdm import tqdm

# %%
y_hats = []

for snapshot in tqdm(test_set):
    visited_nodes = snapshot.x[:,visited_id,:]

    # ids to hex names
    input_trips = [[] for _ in range(visited_nodes.shape[1])]
    for i in range(visited_nodes.shape[0]):
        for j in range(visited_nodes.shape[1]):
            if visited_nodes[i,j]:
                candidate = ids_to_hex_names[i]
                
                if candidate in list(graph.nodes):
                    input_trips[j].append(candidate)
    
    # get distances
    # select 2 nodex with max distance
    source_dest_hexes = []
    for trip in input_trips:
        max_dist_pair = None, None
        max_dist = -1
        for hex_a in trip:
            for hex_b in trip:
                dist = distances_df.at[hex_a,hex_b]
                if max_dist < dist:
                    max_dist = dist
                    max_dist_pair = hex_a, hex_b
        source_dest_hexes.append(max_dist_pair[0])
        source_dest_hexes.append(max_dist_pair[1])
    
    # generate trips
    generated_trips = []
    for _ in range(visited_nodes.shape[1]):
        # makes sure start, end are different
        start, end = random.sample(source_dest_hexes,2)
        trip = nx.shortest_path(graph, start, end)
        trip_hex_ids = [hex_names_to_ids[hex] for hex in trip]
        generated_trips.append(trip_hex_ids)
    
    y_hat = torch.zeros_like(visited_nodes)
    for trip_nr,trip in enumerate(generated_trips):
        for hex_nr in trip:
            y_hat[hex_nr,trip_nr] = 1
        
    y_hats.append(y_hat)

y_hat_baseline = torch.stack(y_hats)

# %%
ns = test_set.snapshot_count
torch.save(y_hat_baseline, f"baselie_res{resolution}_tsin{tsin}_tsout{tsout}_ns{ns}.pt")

# %%
y_hat_baseline.shape


