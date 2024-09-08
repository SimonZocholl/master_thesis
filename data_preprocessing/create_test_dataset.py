# %%
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils

import random
import json
import torch
import pandas as pd
import geopandas as gpd
from tqdm import tqdm

import numpy as np
import networkx as nx

from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.neighbourhoods import H3Neighbourhood
from srai.plotting import plot_regions
import matplotlib.pyplot as plt

from torch_geometric.utils.convert import from_networkx
from torch_geometric_temporal.signal import StaticGraphTemporalSignal
from torch_geometric_temporal.dataset import METRLADatasetLoader

random.seed(3)
%load_ext autoreload
%autoreload 2

# %%
RESOLUTION = 6
NR_PERSONS = 1024
NR_TRAJS_PP = 2*4

# %%
area = geocode_to_region_gdf("Munich, Germany")
regionalizer = H3Regionalizer(resolution=RESOLUTION, buffer=False)

regions = regionalizer.transform(area)
number_hexs = len(regions.index)
rnds = np.random.randint(1, 4, size=(number_hexs,3))
sms = rnds.sum(axis = 0)
r = rnds / sms
regions["work"] = rnds[:,0]
regions["leisure"] = rnds[:,1]
regions["errand"] = rnds[:,2]
regions.head()

# %%
from branca.colormap import linear
colormap = linear.Blues_09.scale(0, 3)

regions.explore("errand", cmap=colormap)

# %%
# TODO: build direction dictionary

def build_edges(regions):
    region_list = regions.reset_index()["region_id"].to_list() # ToDo use a set isntead
    
    adj = np.zeros((len(region_list), len(region_list)))
    
    edge_list = []
    neighbourhood = H3Neighbourhood()
    for i,region in enumerate(region_list):
        neighbours = neighbourhood.get_neighbours_at_distance(region, 1)
        for neighbour in neighbours:
            if neighbour in region_list:
                edge_list.append((region, neighbour))
                edge_list.append((neighbour,region))
                j = region_list.index(neighbour)
                adj[i][j] = 1
                adj[j][i] = 1

    return edge_list, adj


def build_graph(node_features, edge_list):
    graph = nx.Graph()

    for index, node in node_features.iterrows():
        graph.add_node(node.name, hexagon_work=node.work, hexagon_errand=node.errand, hexagon_leisure=node.leisure, hexagon_name=node.name)

    for u,v in edge_list:
        graph.add_edge(u, v)

    return graph

edge_list, adj = build_edges(regions)
graph = build_graph(regions,edge_list)
graph

# %%
pos=nx.spring_layout(graph)
nx.draw(graph,pos, cmap = plt.get_cmap('jet'))
plt.show()

# %%
# generate data

purposes = ["work", "errand", "leisure"]
purposes_to_id = {purpose : i for i, purpose in enumerate(purposes)}

paths = []
trajectory_graphs = []

for i in range(NR_PERSONS):
    personal_values = list(np.random.randint(1,4, size=len(purposes))) #not start with zero to avoid errors
    personal_probabilities = personal_values / sum(personal_values)
    personal = {purpose: probability for purpose, probability in zip(purposes, personal_values)}

    # personalize
    personal_graph = graph.copy()
    nx.set_node_attributes(personal_graph, personal["work"],"personal_work")
    nx.set_node_attributes(personal_graph, personal["errand"],"personal_errand")
    nx.set_node_attributes(personal_graph, personal["leisure"],"personal_leisure")
    nx.set_node_attributes(personal_graph, i,"personal_id")
    
    work_hex = np.random.choice(list(regions.index),1, p=regions["work"]/sum(regions["work"]))[0]
    errand_hex = work_hex
    leisure_hex = work_hex

    while errand_hex == work_hex:
        errand_hex = np.random.choice(list(regions.index),1, p=regions["errand"]/sum(regions["errand"]))[0]
    
    while leisure_hex == work_hex or leisure_hex == errand_hex:
        leisure_hex = np.random.choice(list(regions.index),1, p=regions["leisure"]/sum(regions["leisure"]))[0]
    
    pers_hexs = {"work": work_hex, "errand": errand_hex, "leisure":leisure_hex}

    time = 1
    
    errand_work_path = nx.shortest_path(graph,source=errand_hex,target=work_hex)
    work_leisure_path = nx.shortest_path(graph,source=work_hex,target=leisure_hex)
    leisure_errand_path = nx.shortest_path(graph,source=work_hex,target=leisure_hex)
    
    personal_paths = [["work",errand_work_path], ["leisure",work_leisure_path],["errand",leisure_errand_path]]

    # create paths
    
    '''
    personal_paths = []
    for i in range(NR_TRAJS_PP):
        target = random.choice(purposes)
        source = target

        while source == target:
            source = random.choice(purposes)

        path = nx.shortest_path(graph,source=pers_hexs[source],target=pers_hexs[target])
        personal_paths.append([target, path])
    '''
    
    # process paths
    for purpose, path in personal_paths:

        trajectory_graph = personal_graph.copy()
        nx.set_node_attributes(trajectory_graph, 0,"visited")
        nx.set_node_attributes(trajectory_graph, purposes_to_id[purpose],"purpose")
        nx.set_node_attributes(trajectory_graph, 0,"time")

        for p in path:
            trajectory_graph.nodes[p]["visited"] = 1
            trajectory_graph.nodes[p]["time"] = time
            time += 1

        trajectory_graphs.append(from_networkx(trajectory_graph))
                   

# %%
# graphs to dataset
data = []
for i,g in enumerate(trajectory_graphs):
    d = torch.stack([
                    g.hexagon_work,
                    g.hexagon_errand,
                    g.hexagon_leisure,

                    g.personal_work,
                    g.personal_errand,
                    g.personal_leisure,
                    
                    g.personal_id,

                    g.visited,
                    g.purpose,
                    g.time

                    ], dim=1)
    data.append(d)

data = torch.stack(data)

dataset_dict = {}
pyg_graph = from_networkx(graph)
dataset_dict["edges"] = pyg_graph.edge_index.T.tolist()
dataset_dict["node_ids"] = {name: i for i, name in enumerate(pyg_graph.hexagon_name)}
# (timesteps, nodes, features)
dataset_dict["X"] = data.tolist()


with open("test_dataset.json", "w") as outfile:
    json.dump(dataset_dict, outfile)

# %%
# adjust
# https://pytorch-geometric-temporal.readthedocs.io/en/latest/_modules/torch_geometric_temporal/dataset/metr_la.html#METRLADatasetLoader

class TrajectoryDatasetLoader_Test(object):

    def __init__(self, data_path, trajectories_per_person):
        #self._read_web_data(data_path)
        self.trajectories_per_person = trajectories_per_person
        self._dataset = self._read_web_data(data_path)
        self.node_ids_to_idx, self.idx_to_node_ids = self._get_node_ids_to_idx()
        self._edges = self._get_edges()
        self._edge_weights = self._get_edge_weights()
        self.features_to_idx, self.idx_to_features = self._get_feature_to_idx()
        self.x_features = ["hexagon_work","hexagon_errand", "hexagon_leisure","personal_work","personal_errand","personal_leisure", "visited"]
        self.y_features = ["visited"]

        self.X = np.array(self._dataset["X"])
        self.x, self.y = self._generate_task()
        self.dataset = self.get_dataset()


    def _read_web_data(self,data_path):
        #url = "https://raw.githubusercontent.com/benedekrozemberczki/pytorch_geometric_temporal/master/dataset/chickenpox.json"
        #self._dataset = json.loads(urllib.request.urlopen(url).read())
        
        with open("test_dataset.json") as f:
            _dataset =json.load(f)
        return _dataset
    
    def _get_node_ids_to_idx(self):
        node_ids_to_idx = self._dataset["node_ids"]
        idx_to_node_ids = {v:k for k,v in node_ids_to_idx.items()}
        
        return node_ids_to_idx, idx_to_node_ids
    
    def _get_feature_to_idx(self):
        
        features = ["hexagon_work","hexagon_errand", "hexagon_leisure","personal_work","personal_errand","personal_leisure","personal_id","visited","purpose","time"]
        features_to_idx = {f: i for i,f in enumerate(features)}
        idx_to_features = {i: f for i,f in enumerate(features)}
        return features_to_idx, idx_to_features

    def _get_edges(self):
        return np.array(self._dataset["edges"]).T

    def _get_edge_weights(self):
        return np.ones(self._edges.shape[1])

    def _generate_task(self):

        num_timesteps_in = self.trajectories_per_person // 2
        num_timesteps_out = self.trajectories_per_person - num_timesteps_in
        # (num_timesteps_in, num_nodes, num_node_features) to (num_nodes, num_node_features, num_timesteps_in) 
        X_t = np.transpose(self.X, (1, 2, 0))
        x_features = self.x_features
        y_features = self.y_features
        x_idxs = [self.features_to_idx[x_feature] for x_feature in x_features]
        y_idxs = [self.features_to_idx[y_feature] for y_feature in y_features]
        
        indices = [
            (i, i + (num_timesteps_in + num_timesteps_out))
            for i in range(X_t.shape[2] - (num_timesteps_in + num_timesteps_out) + 1)
        ]

        x,y = [], []
        # (num_nodes, node_features, num_timesteps_in) -> (num_nodes, num_timesteps_out)
        for i,j in indices:
            x.append((X_t[:, x_idxs, i : i + num_timesteps_in]))
            y.append((X_t[:, y_idxs, i + num_timesteps_in : j]))


        x, y = np.array(x), np.array(y)
        return x,y
        
    def get_dataset(self) -> StaticGraphTemporalSignal:
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.x, self.y
        )
        return dataset

trajectory_dataset_loader_test =  TrajectoryDatasetLoader_Test("test_dataset.json", NR_TRAJS_PP)
trajectory_dataset_loader_test.dataset

idx_to_node_ids = trajectory_dataset_loader_test.idx_to_node_ids

# %%
from torch_geometric_temporal.signal import temporal_signal_split

loader = TrajectoryDatasetLoader_Test("test_dataset.json", NR_TRAJS_PP)
dataset = loader.get_dataset()
train_dataset, test_dataset = temporal_signal_split(dataset, train_ratio=0.2)

#train_dataset, test_dataset = dataset, dataset
# examples, nodes, features, timesteps
# ((34249, 207, 2, 12), (34249, 207, 12))

example = next(iter(train_dataset))
example

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
        
    visualize_df = regions.copy()
    visualize_df["count_visited"] = pd.Series(occurance_dict)
    # TODO: add number visited
    
    cmap = linear.Blues_09.scale(0, max(occurance_dict.values()))
    return visualize_df.explore("count_visited", cmap=cmap)

visualize_y(example.y, idx_to_node_ids)

# %%
# https://colab.research.google.com/drive/132hNQ0voOtTVk3I4scbD3lgmPTQub0KR?usp=sharing#scrollTo=7kOvaOrps2oe
# debug and visualize

import torch
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN

# examples, nodes, features, timesteps
class TemporalGNN(torch.nn.Module):
    def __init__(self, node_features, periods):
        super(TemporalGNN, self).__init__()
        # Attention Temporal Graph Convolutional Cell
        self.tgnn = A3TGCN(in_channels=node_features, 
                           out_channels=32, 
                           periods=periods)
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(32, periods)

    def forward(self, x, edge_index):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        
        # TODO: make multivalued predictions possible
        h = self.tgnn(x, edge_index)
        h = F.relu(h)
        h = F.sigmoid(self.linear(h))
        return h

# %%
device = torch.device('cpu')
node_features = 7
periods = NR_TRAJS_PP//2
lr = 0.01
# single example, lr = 0.01, requires 100 epochs for identity

epochs = 50

model = TemporalGNN(node_features=node_features, periods=periods).to(device)
model

# %%
# https://www.kdnuggets.com/2021/09/imbalanced-classification-without-re-balancing-data.html

num_elem = 0
num_ones = 0

for snapshot in tqdm(train_dataset):
    num_elem = num_elem + snapshot.y.numel()
    num_ones = num_ones + snapshot.y.sum()

th = num_ones / num_elem
th

# %%
# TODO: Select appropriate loss function

# Create model and optimizers
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
loss_fn = torch.nn.BCELoss()

model.train()

print("Running training...")
for epoch in range(epochs): 
    loss = 0
    step = 0
    
    for snapshot in tqdm(train_dataset):
        snapshot = snapshot.to(device)
        
        # Get model predictions
        y_hat = model(snapshot.x, snapshot.edge_index).unsqueeze(1)
        loss = loss + loss_fn(y_hat, snapshot.y.float()) 
        step += 1

    loss = loss / (step + 1)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    print("Epoch {} train BCE: {:.4f}".format(epoch, loss.item()))

# %%
# TODO: Evaluate training result
# TODO: Store 

# %%
model.eval()
loss = 0
step = 0

# Store for analysis
predictions = []
labels = []

for snapshot in test_dataset:
    snapshot = snapshot.to(device)
    y_hat = model(snapshot.x, snapshot.edge_index).unsqueeze(1)
    loss = loss + loss_fn(y_hat, snapshot.y.float())
    
    # Store for analysis below
    labels.append(snapshot.y)
    predictions.append(y_hat)
    step += 1

loss = loss / (step+1)
loss = loss.item()
print("Test BCE: {:.4f}".format(loss))

# %%
th

# %%
# Visualization
exampe_id = 3
label = labels[exampe_id]
prediction = predictions[exampe_id]
label, prediction

# %%
# looks strange because order is not preserved
visualize_y(label,idx_to_node_ids)

# %%
visualize_y((prediction.detach()>=th).int(),idx_to_node_ids)

# %%



