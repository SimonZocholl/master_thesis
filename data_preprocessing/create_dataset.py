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
#from torch.utils.data import Dataset, DataLoader
#from torchvision import transforms, utils
import random
import numpy as np
import networkx as nx
import geopandas as gpd
import shapely as sp
import h3 as h3
import pandas as pd

import pandas as pd
from srai.regionalizers import H3Regionalizer, geocode_to_region_gdf
from srai.neighbourhoods import H3Neighbourhood
import matplotlib.pyplot as plt
from torch_geometric.utils.convert import from_networkx

from h3ronpy.arrow.vector import ContainmentMode, wkb_to_cells
import json


#import difftrackgenerator.config as config
from branca.colormap import linear

from shapely.geometry import LineString, Point
import folium

import torch
from torch_geometric_temporal import StaticGraphTemporalSignal

random.seed(3)
# %load_ext autoreload
# %autoreload 2

# %%
# load df

RESOLUTION = 9 #8
FEATURES = ["user_id","visited", "hexagon_work",  "hexagon_leisure", "hexagon_errand", "user_work",  "user_leisure", "user_errand"]

WGS84_EPSG = "EPSG:4326"
METRIC_EPSG = "EPSG:31468"

area = geocode_to_region_gdf("Munich, Germany")

# %%
b_df = pd.read_pickle("../data/osm_buildings_muc.pkl")


# %%
len(b_df)

# %%
# takes >= min
work_set = set(["barn", "stable","kindergarten","fire", "college", "plant","construction","government","conservatory", "university", "diplomatic","consulate", 'farm', 'hospital', "hotel","school", "commercial","greenhouse", "industrial", "works", "aeroway", "office", "military", "amenity ", "craft", "emergency", "healthcare", "shop"])
leisure_set = set(["stable","grandstand", "shrine","religious","civic","synagogue","chapel","church","conservatory", "hotel","stadium", "monastery","aeroway","aerialway","tourism", "sport", "sports", "natural", "leisure", "historic", "amenity", "geological"])
errand_set = set(["warehouse","supermarket", "retail", "healthcare", "shop", "kiosk"])

'''
#work_set = {'agency', 'archaeological', 'art', 'arts', 'atm', 'bank', 'bookmaker', 'building','cannabis', 'carpet', 'casino', 'centre', 'clinic', 'college', 'commercial', 'computer', 'construction', 'consulate', 'copyshop', 'courthouse', 'craft', 'dentist', 'detached', 'diplomatic', 'directors', 'diving', 'doctors', 'driving', 'electrical', 'electronics', 'facility', 'farm', 'fire', 'furniture', 'gallery', 'government', 'hairdresser', 'health', 'hospital', 'hotel', 'industrial', 'kindergarten', 'kitchen', 'laundry', 'library', 'machine', 'mall', 'marketplace', 'medical', 'mobile', 'model', 'museum', 'office', 'outpost', 'pawnbroker', 'pharmacy', 'power', 'prison', 'public', 'radiotechnics', 'rental', 'repair', 'residential', 'restaurant', 'retail', 'riding', 'school', 'scuba', 'security', 'semidetached', 'service', 'services', 'shelter', 'shop', 'station', 'storage', 'supermarket', 'swimming', 'synagogue', 'tailor', 'taxi', 'trailer', 'train', 'transportation', 'university', 'veterinary', 'video', 'warehouse', 'works', 'worship', 'auxiliary', 'barn', 'books', 'castle', 'conservatory', 'court', 'de', 'garage', 'garages', 'motorcycle', 'nursing', 'artwork', 'baby', 'bag', 'bakery', 'bar', 'bath', 'books', 'bowling', 'bungalow', 'castle', 'conservatory', 'court', 'de', 'dog', 'fashion', 'garage', 'garages', 'garden', 'motorcycle', 'nursing', 'rink', 'trophy'}
#leisure_set = {'alley', 'area', 'arts', 'atm', 'attraction', 'bbq', 'beach', 'biergarten', 'bookmaker', 'bridge', 'cafe', 'camp', 'casino', 'centre', 'chapel', 'cinema', 'city', 'civic', 'coffee', 'collector', 'community', 'confectionery', 'courthouse', 'craft', 'cream', 'fountain', 'gallery', 'games', 'golf', 'guest', 'historic', 'hostel', 'leisure', 'mall', 'marketplace', 'massage', 'miniature', 'monastery', 'monument', 'mosque', 'motel', 'museum', 'music', 'musical', 'nature', 'newsagent', 'nightclub', 'park', 'picnic', 'pitch', 'place', 'playground', 'pool', 'recreation', 'restaurant', 'roof', 'sauna', 'scuba', 'seating', 'sports', 'stadium', 'store', 'swimming', 'theatre', 'theme', 'tourism', 'track', 'travel', 'viewpoint', 'works', 'zoo'}
#errand_set = {'accessories', 'aids', 'alcohol', 'amenity', 'antiques', 'apartment', 'apartments', 'appliance', 'bathroom', 'beauty', 'bed', 'beverages', 'bicycle', 'blind', 'boat', 'bookcase', 'boutique', 'box', 'bunker', 'bureau', 'bus', 'butcher', 'cabin', 'camera', 'candles', 'car', 'caravan', 'carport', 'change', 'charging', 'charity', 'cheese', 'chemist', 'childcare', 'chocolate', 'church', 'cleaner', 'cleaning', 'clothes', 'collector', 'container', 'convenience', 'cosmetics', 'course', 'cowshed', 'cream', 'curtain', 'dairy', 'decoration', 'deli', 'department', 'disposal', 'doityourself', 'dormitory', 'drinking', 'dry', 'e-cigarette', 'entrance', 'erotic', 'fabric', 'fast', 'fire', 'fireplace', 'fishing', 'fitness', 'florist', 'food', 'frame', 'fuel', 'funeral', 'furnishing', 'furniture', 'gate', 'general', 'gift', 'glaziery', 'goods', 'grandstand', 'grave', 'greengrocer', 'greenhouse', 'grooming', 'ground', 'hairdresser', 'hall', 'hand', 'hangar', 'hardware', 'hearing', 'herbalist', 'hifi', 'highway', 'home', 'house', 'houseware', 'hunting', 'hut', 'ice', 'information', 'inspection', 'instrument', 'interior', 'internet', 'jewelry', 'kiosk', 'landuse', 'leather', 'library', 'lighting', 'locksmith', 'lottery', 'made', 'man', 'manor', 'medical', 'memorial', 'mobile', 'natural', 'nutrition', 'of', 'optician', 'outdoor', 'paint', 'parking', 'parts', 'party', 'pasta', 'pastry', 'pavilion','perfumery', 'pet', 'phone', 'photo', 'place', 'plant', 'police', 'post', 'presbytery', 'pub', 'pyrotechnics', 'recreation', 'recycling', 'religious', 'reserve', 'rest', 'retail', 'roof', 'sand', 'school', 'seafood', 'second', 'services', 'sewing', 'shed', 'shelter', 'shoes', 'shop', 'shower', 'shrine', 'site', 'slipway', 'social', 'spices', 'spring', 'stable', 'static', 'stationery', 'stop', 'store', 'substation', 'supplements', 'supply', 'synagogue', 'table', 'tattoo', 'tea', 'telephone', 'tent', 'terrace', 'ticket', 'tiles', 'tobacco', 'toilets', 'tower', 'townhall', 'toys', 'trade', 'trailer', 'transformer', 'tyres', 'vacuum', 'variety', 'vehicle', 'vending', 'video', 'wash', 'waste', 'watches', 'water', 'weapons', 'wholesale', 'window', 'wine', 'wool', 'yard', 'yes', 'artwork', 'baby', 'bag', 'bakery', 'bar', 'bath', 'books', 'bowling', 'bungalow', 'castle', 'conservatory', 'court', 'de', 'dog', 'fashion', 'garage', 'garages', 'motorcycle', 'nursing', 'rink', 'trophy'}

b_df = pd.read_pickle("../data/osm_buildings_muc.pkl")
b_gdf = gpd.GeoDataFrame(b_df, geometry="geom", crs=3857).to_crs(crs=WGS84_EPSG)
b_gdf_drop_list = ["area", "mid", "visitor_capacity"]
b_gdf = b_gdf.drop(b_gdf_drop_list, axis=1)

b_gdf["categories"] = b_gdf["full_categories"].apply( lambda x: x.split("_"))
munich_polygon = area["geometry"][0].geoms[0]
b_gdf = b_gdf[b_gdf["geom"].apply(lambda x: munich_polygon.contains(x))]
b_gdf["work"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(work_set))
b_gdf["leisure"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(leisure_set))
b_gdf["errand"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(errand_set))
b_gdf["centroid"] = b_gdf["geom"].apply(sp.centroid)

b_gdf["hex"] = b_gdf["centroid"].apply(lambda x: h3.latlng_to_cell(x.y, x.x, RESOLUTION))

print(len(b_gdf))
b_gdf.head()

# %%
# Get Graph
# Get Hexagon properties

regionalizer = H3Regionalizer(resolution=RESOLUTION, buffer=True)
regions = regionalizer.transform(area)
regions["hexagon_work"] = 0
regions["hexagon_leisure"] = 0
regions["hexagon_errand"] = 0

for region in regions.index:
    sumd_wle = b_gdf[b_gdf["hex"] == region][["work", "leisure", "errand"]].sum()
    regions.at[region, "hexagon_work"] = sumd_wle["work"]
    regions.at[region, "hexagon_leisure"] = sumd_wle["leisure"]
    regions.at[region, "hexagon_errand"] = sumd_wle["errand"]

regions.to_pickle(f"regions_enhanced_res{RESOLUTION}_v2.pkl")

'''
# %%
regions = pd.read_pickle(f"../data/regions_enhanced_res{RESOLUTION}_v2.pkl")

# %%
#visualize = "hexagon_work" # "hexagon_leisure", "hexagon_errand"
#cmap = linear.Blues_09.scale(0, regions[visualize].max())
#regions.explore(visualize,cmap=cmap)

# %%
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

edge_list, adj = build_edges(regions)
base_graph = build_graph(regions,edge_list)


# %%
#pos=nx.kamada_kawai_layout(base_graph)
#nx.draw(base_graph,pos, cmap = plt.get_cmap('jet'), node_size = 7)
#plt.show()

# %%
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

def my_shapely_geometry_to_h3(x):
    wkb = [x.wkb]
    h3_indexes = wkb_to_cells(
            wkb, resolution=RESOLUTION, containment_mode=ContainmentMode.IntersectsBoundary, flatten=True
        ).unique()
    return [h3.int_to_str(h3_index) for h3_index in h3_indexes.tolist()]

# %% [markdown]
# # TEST area
#

# %%
from shapely import LineString
a = LineString([[0, 0], [1, 0], [1, 1]])
a.dtype


# %%
df = pd.read_csv("../data/mydata_raw.csv")


# %%
len(df)

# %%
50034 - 35240 

# %%
droplist = ["Unnamed: 0"]
df = df.drop(droplist, axis=1)
df = df.dropna()
len(df)

# %%
df.head()

# %% [markdown]
# # End of test
#

# %%
# columns have only value one value
# "Unnamed: 0" : 0
# "started_at_timezone": Europe/Berlin
# "finished_at_timezone": Europe/Berlin
# "track" : True
# "mode" : 10
# "purpose": nan
# "updated_at": dont know what it is
df = pd.read_csv("../data/mydata_raw.csv")

droplist = ["Unnamed: 0"]
df = df.drop(droplist, axis=1)
df = df.dropna()
# remove multi line strings (130 occurances)
df = df[~df['geom'].str.contains("MULTILINESTRING")]

# parse time information
# https://jeffkayser.com/projects/date-format-string-composer/index.html
#df["started_at"] = pd.to_datetime(df["started_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)
#df["finished_at"] = pd.to_datetime(df["finished_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)

# parse lineStrings
df["geom"] = df["geom"].apply(string_to_lineString)
df["age"] = df["age"].astype(int)

survery_columns = ['user_id','gender', 'age', 'income_net', 'job', 'education', 'hh_size', 'hh_children', 'usage_car', 'activity_work', 'activity_leisure', 'car', 'activity_errand', 'distance_work', 'distance_leisure', 'distance_errand']
survey_df =  df[survery_columns].drop_duplicates()
area = geocode_to_region_gdf("Munich, Germany")
gdf = gpd.GeoDataFrame(df, crs=WGS84_EPSG,geometry="geom")
gdf["polyline"] = gdf["geom"].apply(linestring_to_polyline)
gdf["h3_hex"] = gdf["geom"].apply(lambda x: my_shapely_geometry_to_h3(x))

#gdf["start"] = gdf["geom"].apply(lambda x: Point(x.coords[0]))
#gdf["end"] = gdf["geom"].apply(lambda x: Point(x.coords[-1]))
#gdf["start_h3_hex"] = gdf["start"].apply(lambda x: my_shapely_geometry_to_h3(x))
#gdf["end_h3_hex"] = gdf["end"].apply(lambda x: my_shapely_geometry_to_h3(x))
#gdf["started_at_ts"] = gdf["started_at"].apply(pd.Timestamp.timestamp)
#gdf["finished_at_ts"] = gdf["finished_at"].apply(pd.Timestamp.timestamp)
#e_gdf = pd.read_csv("../data/expensive_gdf.csv")
#gdf["osm_end"] = e_gdf["osm_end"]

gdf = gdf.dropna()
#gdf.head()

# %%
# get count trajectories
# filter all users with user.TRAJECTORY_COUNT < TRAJECTORY_COUNT
# first attempt get one source target sample out of all remaining persons
# store trajectory, person info on graph
users = gdf["user_id"].unique()
user_df = pd.DataFrame(columns=["user_id", "user_work", "user_errand", "user_leisure", "trajectories"])
purposes = ["work", "leisure", "errand"]

for i,user in enumerate(users):
    
    user_dict = dict()
    entries = gdf[gdf["user_id"] == user]
    trajectories = []
    for key, val in entries.iterrows():
        trajectories.append([hex for hex in val["h3_hex"]])
    

    user_dict["user_id"] = user
    user_dict["user_work"] = entries["activity_work"].values[0]
    user_dict["user_errand"] = entries["activity_errand"].values[0]
    user_dict["user_leisure"] = entries["activity_leisure"].values[0]
    user_dict["trajectories"] = [trajectories]

    user_df = pd.concat([user_df, pd.DataFrame(user_dict)], ignore_index=True)

# %%
pyg_graph = from_networkx(base_graph)

activity_to_idx = {'Never': 0,'Once a week': 1,'Twice a week': 2,
                   '3 times a week' : 3,'4 times a week': 4,'5 times a week': 5,'more than 5 times a week':6}

user_id_to_ids = { uid: i for i, uid in enumerate(users)}
features_to_ids = {f: i for i,f in enumerate(FEATURES)}
purposes_to_ids = {purpose: i for i, purpose in enumerate(purposes)}
hex_names_to_ids = {name: i for i, name in enumerate(pyg_graph.hex_name)}

# %%
from tqdm import tqdm

# %%
## IDEA

# takes >=12 mins
# higher resolution >= 70min
graphs = []
skipped = 0
not_skipped = 0
edge_usage = {f"{i},{j}" :0 for i,j in from_networkx(base_graph).edge_index.T}


for i, d in tqdm(user_df.iterrows()):
    personal_graph = base_graph.copy()

    nx.set_node_attributes(personal_graph, activity_to_idx[d["user_work"]],"user_work")
    nx.set_node_attributes(personal_graph, activity_to_idx[d["user_leisure"]],"user_leisure")
    nx.set_node_attributes(personal_graph, activity_to_idx[d["user_errand"]],"user_errand")
    nx.set_node_attributes(personal_graph, user_id_to_ids[d["user_id"]],"user_id")
    
    trajectory_graph = personal_graph.copy()    
    for trajectory in d["trajectories"]:
        if any(hex not in trajectory_graph.nodes for hex in trajectory ):
            skipped += 1
            continue
        
        not_skipped += 1
        trajectory_graph = personal_graph.copy()
        nx.set_node_attributes(trajectory_graph, 0,"visited")

        old_hex = None
        for hex in trajectory:
            trajectory_graph.nodes[hex]["visited"] = 1
            if old_hex is not None:
                i = hex_names_to_ids[old_hex]
                j = hex_names_to_ids[hex]
                if f"{i},{j}" in edge_usage.keys():
                    edge_usage[f"{i},{j}"] += 1

            old_hex = hex
        graphs.append(from_networkx(trajectory_graph))

#len(graphs), skipped,not_skipped

# %%
# graphs to dataset
data = []
for i,g in enumerate(graphs):
    # must match "FEATURES" order        
    d = torch.stack([g.user_id,
                     g.visited,
                     g.hexagon_work,
                     g.hexagon_leisure,
                     g.hexagon_errand,
                     g.user_work,
                     g.user_leisure,
                     g.user_errand
                    ], dim=1)
    data.append(d)
data = torch.stack(data)

dataset_dict = {}
dataset_dict["edges"] = pyg_graph.edge_index.T.tolist()


# Lookup tables
dataset_dict["resolution"] = RESOLUTION
dataset_dict["features"] = FEATURES

dataset_dict["hex_names_to_ids"] = {name: i for i, name in enumerate(pyg_graph.hex_name)}
dataset_dict["activity_to_ids"] = activity_to_idx
dataset_dict["user_id_to_ids"] = user_id_to_ids
dataset_dict["features_to_ids"] = features_to_ids
dataset_dict["edge_usages"] = list(edge_usage.values())

# (timesteps, nodes, features)
dataset_dict["X"] = data.tolist()

# %%
name = f"dataset_res{RESOLUTION}.json"
with open(name, "w") as outfile:
    json.dump(dataset_dict, outfile)


