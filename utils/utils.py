import torch
import numpy as np
import pandas as pd
import folium 
import json
import h3
import os
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import branca.colormap as cm

from sklearn.metrics import auc
from sklearn.metrics import precision_recall_curve
from folium.plugins import DualMap
from branca.colormap import linear
from geopy.distance import geodesic
from pyproj import Transformer
from shapely import MultiPoint, Point
from collections import Counter
from shapely.geometry import LineString
from h3ronpy.arrow.vector import ContainmentMode, wkb_to_cells
from tqdm import tqdm
from srai.neighbourhoods import H3Neighbourhood
import networkx as nx

def test_loop(model, test_transform, test_dataset, device):
    model.to(device)
       
    # Store for analysis
    predictions = []
    labels = []

    model.eval()
    snapshot = next(iter(test_dataset))
    edge_index = snapshot.edge_index.to(device)
    edge_attr = snapshot.edge_attr.to(device)

    for snapshot in test_dataset:
        with torch.no_grad():
            x = snapshot.x
            y = snapshot.y.float().to(device)
            x = test_transform.min_max_scale(x).float().to(device)
            y_hat = model(x, edge_index, edge_attr)

            # Store for analysis below
            predictions.append(y_hat)
            labels.append(y)

    return predictions, labels

def train_loop(model_name, model, criterion, optimizer, device, epochs, train_transform, test_transform, train_dataset,test_dataset):
    
    # Setup
    model.to(device)    
    model_paths = []

    # Training
    snapshot = next(iter(train_dataset))
    edge_index = snapshot.edge_index.to(device)
    edge_attr = snapshot.edge_attr.to(device)
    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)):

        train_loss = 0
        # Training
        model.train()
        
        for snapshot in train_dataset:
            
            x = snapshot.x
            y = snapshot.y.float().to(device)
            x = train_transform.min_max_scale(x).float().to(device)            
            y_hat = model(x, edge_index,edge_attr)
            train_loss = train_loss + criterion(y_hat, y)

        train_loss = train_loss/train_dataset.snapshot_count
        train_loss.backward()
        optimizer.step()    
        optimizer.zero_grad()
        
        #print("Epoch {} train BCE: {:.4f}".format(epoch, train_loss.item()))

        # saving
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()        
            }
        checkpoint_dir = os.path.join(os.curdir, "checkpoints",model_name)
        model_path = os.path.join(checkpoint_dir,"epoch"+str(epoch)+".pth")
        model_paths.append(model_path)

        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save(checkpoint_data, model_path)
        
        model.eval()
        train_loss = 0
        for snapshot in train_dataset:
            with torch.no_grad():
                x = snapshot.x
                y = snapshot.y.float().to(device)
                x = train_transform.min_max_scale(x).float().to(device)
                y_hat = model(x, edge_index, edge_attr)
                train_loss = train_loss + criterion(y_hat, y)
               
        train_loss = train_loss / train_dataset.snapshot_count
        train_losses.append(train_loss.item())

        val_loss = 0
        for snapshot in test_dataset:
            with torch.no_grad():
                x = snapshot.x
                y = snapshot.y.float().to(device)
                x = test_transform.min_max_scale(x).float().to(device)
                y_hat = model(x, edge_index, edge_attr)
                val_loss = val_loss + criterion(y_hat, y)
        
        val_loss = val_loss / test_dataset.snapshot_count
        val_losses.append(val_loss.item())


    # get best model, based on training losses
    best_model_index = np.argmin(train_losses)
    checkpoint = torch.load(model_paths[best_model_index])
    checkpoint_data = {
            "epoch": best_model_index,
            "model_state_dict": checkpoint["model_state_dict"],
            "optimizer_state_dict": optimizer.state_dict()        
            }
    model_dir = os.path.join(os.curdir, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir,model_name +".pth")
    torch.save(checkpoint_data, model_path)
    for path in model_paths:
        os.remove(path)
        
    return model, train_losses, val_losses



def get_threshold_and_fscore(labels, predictions):
    labels_arr_flatt = torch.stack(labels).flatten().detach().numpy()
    predictions_arr_flatt = torch.stack(predictions).flatten().detach().numpy()
    
    precision, recall, thresholds =precision_recall_curve(labels_arr_flatt, predictions_arr_flatt)
    # convert to f score
    epsilon = 1e-10*np.ones_like(precision)
    fscore = (2 * precision * recall) / (precision + recall +epsilon)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    best_threshold =  thresholds[ix]
    best_fscore = fscore[ix]
    auc_score = auc(recall, precision)
    return best_threshold, best_fscore, auc_score


def get_pos_weights(train_set):
    num_elem = torch.sum(torch.tensor([snapshot.y.numel() for snapshot in train_set]))
    num_ones = torch.sum(torch.tensor([snapshot.y.sum() for snapshot in train_set]))
    pos_weight = (num_elem- num_ones)/num_ones
    return pos_weight


def add_data_to_json(filename, new_data):
    # Load existing JSON data
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Add new data
    data.update(new_data)
    
    # Write updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f)


def create_json_file(filename):
    # Create an empty JSON file
    with open(filename, 'w') as f:
        json.dump({}, f)

def string_to_lineString(s: str):
    coords_str = s[12:-1]
    coords = [[float(strng[0]),float(strng[1])] for strng in (pair.split() for pair in coords_str.split(","))]
    return LineString(coords)

## !!! Turn arround lat and lon information !!!
# WGS84_EPSG = "EPSG:4326"
# METRIC_EPSG = "EPSG:31468"
def linestring_to_polyline(linestring):
    lons, lats    = linestring.xy
    loc = [[lat,lon] for lat,lon in zip(lats,lons)]
    res = folium.PolyLine(loc)
    return res

def my_shapely_geometry_to_h3(x, RESOLUTION):
    wkb = [x.wkb]
    h3_indexes = wkb_to_cells(
            wkb, resolution=RESOLUTION, containment_mode=ContainmentMode.IntersectsBoundary, flatten=True
        ).unique()
    return [h3.int_to_str(h3_index) for h3_index in h3_indexes.tolist()]

# somewhat buggy
def visualize_single_prediction_label_pairs(prediction_t, label_t, ids_to_hex_names, regions):

    visualize_df = regions.copy()
    for i in range(label_t.shape[0]):
        visualize_df[f"label{i}"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(label_t[i,:].int()) }
        visualize_df[f"prediction{i}"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(prediction_t[i,:].int()) }
       
    cmap = cm.StepColormap(["white", "blue"], vmin=0, vmax=1) 
    dual_m = DualMap(location=(48.12, 11.58), zoom_start=10)
    #folium.TileLayer("openstreetmap").add_to(dual_m)

    for i in range(label_t.shape[0]):
        visualize_df.explore(name=f"prediction{i}", m=dual_m.m2, cmap=cmap, column=f"prediction{i}", style_kwds = {"fillOpacity" : 0.2})
        visualize_df.explore(name=f"label{i}",     m=dual_m.m1, cmap=cmap, column=f"label{i}",       style_kwds = {"fillOpacity" : 0.2},)
        
    folium.LayerControl().add_to(dual_m)
    return dual_m

def visualize_all_prediction_label_pairs(label, prediction, ids_to_hex_names, regions):

    visualize_df = regions.copy()
    visualize_df["labels"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(label.int()) }
    visualize_df["predictions"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(prediction.int()) }
    max_visits_labels = visualize_df["labels"].quantile(0.9) # max()

    max_visits_predictions = visualize_df["predictions"].quantile(0.9) # max()

    cmap_labels = cm.LinearColormap(["white", "blue"], vmin=0, vmax=max_visits_labels)
    cmap_preds = cm.LinearColormap(["white", "blue"], vmin=0, vmax=max_visits_predictions)

    dual_m = DualMap(location=(48.12, 11.58), tiles=None, zoom_start=10)
    folium.TileLayer("openstreetmap").add_to(dual_m)
    visualize_df.explore(name=f"labels",     m=dual_m.m2, cmap=cmap_labels, column=f"labels",       style_kwds = {"fillOpacity" : 0.8},)
    visualize_df.explore(name=f"predictions", m=dual_m.m1, cmap=cmap_preds, column=f"predictions", style_kwds = {"fillOpacity" : 0.8})
    
    folium.LayerControl().add_to(dual_m)
    return dual_m



def get_normalized_visited_difference(labelss, predictionss):
    return (torch.sum(predictionss) -torch.sum(labelss))  / torch.sum(labelss)


def abs_visited_difference(prediciton, label):
    approx_travel_dist_label = torch.sum(label)
    approx_travel_dist_pred = torch.sum(prediciton)
    return torch.abs(approx_travel_dist_label-approx_travel_dist_pred)

def get_threshold_and_fscore(labels, predictions):
    labels_arr_flatt = labels.flatten()
    predictions_arr_flatt = predictions.flatten()
    
    precision, recall, thresholds =precision_recall_curve(labels_arr_flatt, predictions_arr_flatt)
    # convert to f score
    epsilon = 1e-10*np.ones_like(precision)
    fscore = (2 * precision * recall) / (precision + recall +epsilon)
    # locate the index of the largest f score
    ix = np.argmax(fscore)
    best_threshold =  thresholds[ix]
    best_fscore = fscore[ix]
    auc_score = auc(recall, precision)
    return best_threshold, best_fscore, auc_score

def intersection_over_union(labelss, predictionss):
    predictionss_visits_per_node =torch.sum(predictionss, dim=(0,2))
    labelss_visits_per_node =torch.sum(labelss, dim=(0,2))

    max_per_node = torch.max(predictionss_visits_per_node, labelss_visits_per_node)
    min_per_node = torch.min(predictionss_visits_per_node, labelss_visits_per_node)
    iou = torch.sum(min_per_node) / torch.sum(max_per_node)

    return iou


def create_json_file(filename):
    # Create an empty JSON file
    with open(filename, 'w') as f:
        json.dump({}, f)

def add_data_to_json(filename, new_data):
    # Load existing JSON data
    with open(filename, 'r') as f:
        data = json.load(f)
    
    # Add new data
    data.update(new_data)
    
    # Write updated data back to the file
    with open(filename, 'w') as f:
        json.dump(data, f)


def ellipsoidal_distance(p1, p2) -> float:
    """ Calculate distance (in meters) between p1 and p2, where 
    each point is represented as a tuple (lat, lon) """
    return geodesic(p1, p2).meters

def compute_distance_matrix(df_sites, dist_metric=ellipsoidal_distance):
    """ Creates an N x N distance matrix from a dataframe of N locations 
    with a latitute column and a longitude column """
    df_dist_matrix = pd.DataFrame(index=df_sites.index, columns=df_sites.index)

    for orig, orig_loc in df_sites.iterrows():  # for each origin
        for dest, dest_loc in df_sites.iterrows():  # for each destination
            df_dist_matrix.at[orig, dest] = dist_metric(orig_loc, dest_loc)
    return df_dist_matrix

def get_distances_df(regions):
    regions["lat"] = regions.index.to_series().apply(lambda x: h3.cell_to_latlng(x)[0])
    regions["lon"] = regions.index.to_series().apply(lambda x: h3.cell_to_latlng(x)[1])
    distances_df = compute_distance_matrix(regions[["lat", "lon"]])
    return distances_df


def radius_of_gyration(ys,ids_to_hex_names, distances_df):
    METRIC_EPSG = "EPSG:31468"
    WGS84_EPSG = "EPSG:4326" 
    
    trips = []
    for y in ys:
        for j in range(y.shape[1]):
            trip = [ids_to_hex_names[i] for i in range(y.shape[0]) if y[i,j]]
            if trip:
                trips.append(trip)

    source_dest_hexes = []
    # get distances
    # select 2 nodex with max distance
    for trip in trips:
        max_dist_pair = None, None
        max_dist = -1
        for hex_a in trip:
            for hex_b in trip:
                dist = distances_df.at[hex_a,hex_b]
                if max_dist < dist:
                    max_dist = dist
                    max_dist_pair = hex_a, hex_b
        if max_dist_pair is not None:
            source_dest_hexes.append(max_dist_pair[0])
            source_dest_hexes.append(max_dist_pair[1])

    transformer = Transformer.from_crs(WGS84_EPSG, METRIC_EPSG)
    source_dest_points_t = ([transformer.transform(*h3.cell_to_latlng(hex)) for hex in source_dest_hexes])
    points_t = MultiPoint(source_dest_points_t)
    c = points_t.centroid

    k = len(source_dest_points_t)

    # catch errors
    if k == 0:
        return 0
    rg = np.sqrt(1/k * np.sum([c.distance(p)**2 for p in points_t.geoms]))
    return rg

def top_k_frequent_elements(lst, k):
    counts = Counter(lst)
    top_k = counts.most_common(k)
    return [element for element, _ in top_k]


def k_radius_of_gyration(ys,ids_to_hex_names, distances_df,k=2):
    METRIC_EPSG = "EPSG:31468"
    WGS84_EPSG = "EPSG:4326"

    
    trips = []
    for y in ys:
        for j in range(y.shape[1]):
            trip = [ids_to_hex_names[i] for i in range(y.shape[0]) if y[i,j]]
            if trip:
                trips.append(trip)

    source_dest_hexes = []

    # get distances
    # select 2 nodex with max distance
    for trip in trips:
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

    top_k_source_dest_hexes = top_k_frequent_elements(source_dest_hexes, k)
    # TODO make sure
    METRIC_EPSG = "EPSG:31468"
    transformer = Transformer.from_crs(WGS84_EPSG, METRIC_EPSG)    
    top_k_source_dest_points_t = ([Point(transformer.transform(*h3.cell_to_latlng(hex))) for hex in top_k_source_dest_hexes])
    centroid_k = MultiPoint(top_k_source_dest_points_t).centroid

    k = len(top_k_source_dest_points_t)
    if k == 0:
        return 0
    
    rg = np.sqrt(1/k * np.sum([centroid_k.distance(p)**2 for p in top_k_source_dest_points_t]))
    return rg      

# jump size
def get_jump_sizes(y,ids_to_hex_names, distances_df):

    # trip, nodes
    trips = [[] for _ in range(y.shape[1])]
    for i in range(y.shape[0]):
        for j in range(y.shape[1]):
            if y[i,j]:
                trips[j].append(ids_to_hex_names[i])

    # get distances
    # select 2 nodex with max distance
    jump_sizes = []
    for trip in trips:
        max_dist = -1
        for hex_a in trip:
            for hex_b in trip:
                dist = distances_df.at[hex_a,hex_b]
                if max_dist < dist:
                    max_dist = dist
        
        jump_sizes.append(max_dist)
    return jump_sizes

def load_from_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


def build_edges(regions):
    region_list = regions.reset_index()["region_id"].to_list()
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
        graph.add_node(node.name, hexagon_work=node.hexagon_work, hexagon_errand=node.hexagon_errand, hexagon_leisure=node.hexagon_leisure, hex_name = node.name)
    
    for u,v in edge_list:
        graph.add_edge(u, v)
    return graph

# https://stackoverflow.com/questions/38649501/labeling-boxplot-in-seaborn-with-median-value
def add_median_labels(ax: plt.Axes, fmt: str = ".2f", fontsize=8) -> None:
    """Add text labels to the median lines of a seaborn boxplot.

    Args:
        ax: plt.Axes, e.g. the return value of sns.boxplot()
        fmt: format string for the median value
    """
    lines = ax.get_lines()
    boxes = [c for c in ax.get_children() if "Patch" in str(c)]
    start = 4
    if not boxes:  # seaborn v0.13 => fill=False => no patches => +1 line
        boxes = [c for c in ax.get_lines() if len(c.get_xdata()) == 5]
        start += 1
    lines_per_box = len(lines) // len(boxes)
    for median in lines[start::lines_per_box]:
        x, y = (data.mean() for data in median.get_data())
        # choose value depending on horizontal or vertical plot orientation
        value = x if len(set(median.get_xdata())) == 1 else y
        text = ax.text(x, y, f'{value:{fmt}}', ha='center', va='center', color='white', fontsize = fontsize)
        # create median-colored border around white text for contrast
        text.set_path_effects([
            path_effects.Stroke(linewidth=3, foreground=median.get_color()),
            path_effects.Normal(),
        ])