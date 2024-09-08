# %%
import sys
sys.path.append('../')
sys.path.append('../../')
import networkx as nx
from collections import Counter
import folium
from srai.plotting import plot_regions
import matplotlib.pyplot as plt
import pandas as pd
import geopandas as gpd
import numpy as np
import seaborn as sns
from tqdm import tqdm
import torch
from pyproj import Transformer
from srai.regionalizers import  geocode_to_region_gdf
from shapely.geometry import  Point, MultiPoint
import h3
from iteration_utilities import flatten
from srai.regionalizers import H3Regionalizer
from branca.colormap import linear
import shapely as sp
from master_thesis.utils.utils import build_edges, build_graph, get_distances_df

from master_thesis.utils.utils import radius_of_gyration
from master_thesis.utils.utils import k_radius_of_gyration
from master_thesis.utils.utils import get_jump_sizes
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.utils import string_to_lineString
from master_thesis.utils.utils import my_shapely_geometry_to_h3
from master_thesis.utils.utils import top_k_frequent_elements

%load_ext autoreload
%autoreload 2

# %%
# TODO: set fig sizes !!!
figsize_small=(10.5/2.54, 7.25/2.54)
figsize_large=(15.5/2.54, 10/2.54)
figsize_medium=(15.5/2.54, 7/2.54)

fontsize_small = 6
fontsize_medium = 8
fontsize_large = 12

dpi=1000

# %%
WGS84_EPSG = "EPSG:4326"
METRIC_EPSG = "EPSG:31468"
# 3857
RESOLUTION = 8

res8 = 8
res9 = 9
res7 = 7

# %%
input_steps = 3
output_steps = 3

loader = TrajectoryDatasetLoader(f"../data/dataset_res{RESOLUTION}_v2.json", input_steps, output_steps)
dataset = loader.get_dataset()

ids_to_hex_names = loader.ids_to_hex_names
ids_to_features = loader.ids_to_features
features_to_ids = loader.features_to_ids
ids_to_activity = loader.ids_to_activity
sample_to_user_id = loader.sample_to_user_id

# %%
# ["hexagon_work", "hexagon_leisure", "hexagon_errand", "user_work", "user_errand", "user_leisure", "visited"]

work_id = 3  #correct for user_id missing and messup
leisure_id = 5 #correct for user_id missing
errand_id = 4  #correct for user_id missing
visited_id = 6 #correct for user_id missing

# %%
# argumentation for above
x_features = ["hexagon_work","hexagon_leisure","hexagon_errand", "user_work","user_errand","user_leisure", "visited"]
x_ids = [features_to_ids[x_feature] for x_feature in x_features]
["hexagon_work", "hexagon_leisure", "hexagon_errand", "user_work", "user_errand", "user_leisure", "visited"]
x_ids

# %%
next(iter(dataset))

# %%
regions = pd.read_pickle(f"../data/regions_enhanced_res{RESOLUTION}_v2.pkl")
distances_df = get_distances_df(regions)
regions.head()

# %%
ss = next(iter(dataset))

torch.concat([ss.x[:,visited_id,:], ss.y], axis=1).shape

# %%
processed_all_trips_tensor = torch.stack([ torch.concat([ss.x[:,visited_id,:], ss.y], axis=1) for ss in dataset])

processed_wle_tensor = torch.stack([ snapshot.x[0,[work_id,leisure_id, errand_id],0] for snapshot in dataset])
processed_wle_tensor.shape, processed_all_trips_tensor.shape

# %%
processed_wle_dict = dict()
processed_wle_dict["activity_work"] = [ ids_to_activity[w.item()]for w in processed_wle_tensor[:,0]]
processed_wle_dict["activity_leisure"] = [ ids_to_activity[l.item()]for l in processed_wle_tensor[:,1]]
processed_wle_dict["activity_errand"] = [ ids_to_activity[e.item()]for e in processed_wle_tensor[:,2]]
processed_wle_df = pd.DataFrame(processed_wle_dict)
processed_wle_df.head()

# %% [markdown]
# # Region

# %%
region_muc = geocode_to_region_gdf("Munich, Germany").to_crs(METRIC_EPSG)
region_muc.explore()

# %%
int(region_muc["geometry"].area[0])

# %%
# https://wolf-h3-viewer.glitch.me/

regionalizer = H3Regionalizer(resolution=RESOLUTION, buffer=True)
regionalizer7 = H3Regionalizer(resolution=res7, buffer=True)
regionalizer8 = H3Regionalizer(resolution=res8, buffer=True)
regionalizer9 = H3Regionalizer(resolution=res9, buffer=True)

regions7 = regionalizer7.transform(region_muc).to_crs(METRIC_EPSG)
regions8 = regionalizer8.transform(region_muc).to_crs(METRIC_EPSG)
regions9 = regionalizer9.transform(region_muc).to_crs(METRIC_EPSG)
regions789 = [regions7, regions8, regions9]

# %%
regions7.explore()

# %%
regions8.explore()

# %%
# https://towardsdatascience.com/exploring-location-data-using-a-hexagon-grid-3509b68b04a2

num_hexs =[len(region) for region in regions789]
area_all_hexs =[int(region["geometry"].area.sum()) for region in regions789]
avg_area_hexs =[int(region["geometry"].area.sum()/len(region)) for region in regions789]
analog_size_hexs = ["campus","park","stadium"]

regions_data_df = pd.DataFrame({"resolution": [7,8,9], "num_hexes": num_hexs, "area_all_hexs": area_all_hexs, "avg_area_hexs": avg_area_hexs, "analog_size_hexs" : analog_size_hexs})
regions_data_df

# %% [markdown]
# # Points Of Interest

# %%
work_set = set(["barn", "stable","kindergarten","fire", "college", "plant","construction","government","conservatory", "university", "diplomatic","consulate", 'farm', 'hospital', "hotel","school", "commercial","greenhouse", "industrial", "works", "aeroway", "office", "military", "amenity ", "craft", "emergency", "healthcare", "shop"])
leisure_set = set(["stable","grandstand", "shrine","religious","civic","synagogue","chapel","church","conservatory", "hotel","stadium", "monastery","aeroway","aerialway","tourism", "sport", "sports", "natural", "leisure", "historic", "amenity", "geological"])
errand_set = set(["warehouse","supermarket", "retail", "healthcare", "shop", "kiosk"])
poi_processed_list = list(work_set.union(leisure_set).union(errand_set))

poi_porcessed_df = pd.DataFrame({"label": poi_processed_list})
poi_porcessed_df.head()
poi_porcessed_df["work"] = poi_porcessed_df["label"].apply(lambda x : x in work_set)
poi_porcessed_df["leisure"] = poi_porcessed_df["label"].apply(lambda x : x in leisure_set)
poi_porcessed_df["errand"] = poi_porcessed_df["label"].apply(lambda x : x in errand_set)
#print(poi_porcessed_df.to_latex())

# %%
b_df = pd.read_pickle("../data/osm_buildings_muc.pkl")
b_gdf = gpd.GeoDataFrame(b_df, geometry="geom", crs=3857).to_crs(crs=WGS84_EPSG)

# %%
b_gdf_drop_list = ["area", "mid", "visitor_capacity"]
b_gdf = b_gdf.drop(b_gdf_drop_list, axis=1)
b_gdf["categories"] = b_gdf["full_categories"].apply( lambda x: x.split("_"))
b_gdf["work"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(work_set))
b_gdf["leisure"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(leisure_set))
b_gdf["errand"] = b_gdf["categories"].apply( lambda x: not set(x).isdisjoint(errand_set))

region_muc = region_muc.to_crs(crs=WGS84_EPSG)
munich_polygon = region_muc["geometry"][0].geoms[0]
b_gdf = b_gdf[b_gdf["geom"].apply(lambda x: munich_polygon.contains(x))]
b_gdf["centroid"] = b_gdf["geom"].apply(sp.centroid)

b_gdf["hex"] = b_gdf["centroid"].apply(lambda x: h3.latlng_to_cell(x.y, x.x, RESOLUTION))

# %%
b_gdf.head()

# %%
# TODO
folium_map = plot_regions(region_muc, colormap=["rgba(0,0,0,0.1)"])

tooltip = folium.GeoJsonTooltip(
    fields=["osm_id", "work","leisure", "errand"],
    labels=True,
)

folium.GeoJson(b_gdf[["geom","osm_id", "work","leisure", "errand"]], tooltip=tooltip, style_function=(lambda x: {"weight": 1})).add_to(folium_map)
folium_map.save("all_pois.html")

# %%
mapped_b_gdf = b_gdf[b_gdf["work"] | b_gdf["leisure"] | b_gdf["errand"]]

num_POIs = len(b_gdf)
num_unique_categories = len(b_gdf["full_categories"].unique())
poi_area = int(b_gdf["geom"].to_crs(METRIC_EPSG).area.sum())

"num_POIs: ", num_POIs, "num_unique_categories: ", num_unique_categories, "poi_area: ", poi_area

# %%
num_mapped_POIs = len(mapped_b_gdf)
num_mapped_unique_POIs = 3
poi_area = int(b_gdf.at[0,"geom"].area)
poi_mapped_area = int(mapped_b_gdf["geom"].to_crs(METRIC_EPSG).area.sum())


"num_mapped_POIs: ",num_mapped_POIs, "num_unique_categories: " ,3, num_mapped_POIs, "poi_mapped_area: ", poi_mapped_area

# %%
poi_original_summary_df =pd.DataFrame({"POIs": [204003], "Labels": [330], "Area Covered": [143794463], "Area Coverd Ratio": [143794463/310721232]})
poi_original_summary_df["Source"] = "Original"
poi_processed_summary_df =pd.DataFrame({"POIs": [41263], "Labels": [3], "Area Covered": [poi_mapped_area], "Area Coverd Ratio": [poi_mapped_area/310721232]})
poi_processed_summary_df["Source"] = "Processed"

poi_summary_df = pd.concat([poi_original_summary_df, poi_processed_summary_df])
print(poi_summary_df.to_latex())

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

ax0 = b_gdf['full_categories'].value_counts().nlargest(5).plot(kind="bar", xlabel="", title="top5 original labels", ax=axes[0], fontsize=fontsize_medium)
ax0.bar_label(ax0.containers[0], fontsize=fontsize_medium)
axes[0].title.set_size(fontsize_medium)

ax1 = mapped_b_gdf[["work","leisure","errand"]].sum().plot(kind="bar", ax=axes[1], title="processed labels", fontsize=fontsize_medium)
ax1.bar_label(ax1.containers[0], fontsize=fontsize_medium)
axes[1].title.set_size(fontsize_medium)

axes[0].set_xlabel("Top-5 Labels Original", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("All 3 Labels Processed", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='x',  labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)


axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.1)
axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.1)

xticklabels = ["yes", "appartments","garrage","house","reidential"]
ax0.set_xticklabels(xticklabels)

plt.tight_layout()

#fig.savefig('../plots/evaluation/labels_compare_bar.png',dpi=dpi )

# %%
# geammt müncehn 310721232
# gesammt POI 143794463
# pp POI 108830639

108830639 / 143794463, 41264 /204003

# %%
regionalizer = H3Regionalizer(resolution=RESOLUTION, buffer=True)
regions_hex = regionalizer.transform(regions8).to_crs(WGS84_EPSG)
regions_hex["hexagon_work"] = 0
regions_hex["hexagon_leisure"] = 0
regions_hex["hexagon_errand"] = 0

for region in regions_hex.index:
    sumd_wle = b_gdf[b_gdf["hex"] == region][["work", "leisure", "errand"]].sum()
    regions_hex.at[region, "hexagon_work"] = sumd_wle["work"]
    regions_hex.at[region, "hexagon_leisure"] = sumd_wle["leisure"]
    regions_hex.at[region, "hexagon_errand"] = sumd_wle["errand"]

# %%
regions_hex.head()

# %%
visualize = "hexagon_errand" # "hexagon_work", "hexagon_leisure", "hexagon_errand"

# %%
# nice to have
regions_hex["hexagon_work"].sort_values().plot(kind="line")

# %%

cmap = linear.Blues_09.scale(0, regions_hex[visualize].quantile(q=0.9))
maap = regions_hex.explore(visualize,cmap=cmap)
maap

# %%
# TODO: plot what is most important in hex

# %% [markdown]
# # Personal 

# %%
df = pd.read_csv("../data/mydata_raw.csv")

droplist = ["Unnamed: 0"]
df = df.drop(droplist, axis=1)
df = df.dropna()
# remove multi line strings (130 occurances)
df = df[~df['geom'].str.contains("MULTILINESTRING")]

# parse lineStrings
df["geom"] = df["geom"].apply(string_to_lineString)
df["age"] = df["age"].astype(int)

area = geocode_to_region_gdf("Munich, Germany")
gdf = gpd.GeoDataFrame(df, crs=WGS84_EPSG,geometry="geom")
gdf["h3_hex"] = gdf["geom"].apply(lambda x: my_shapely_geometry_to_h3(x, RESOLUTION))

gdf = gdf.to_crs(METRIC_EPSG)
gdf["start"] = gdf["geom"].apply(lambda x: Point(x.coords[0]))
gdf["end"] = gdf["geom"].apply(lambda x: Point(x.coords[-1]))

gdf = gdf.dropna()

# %%
gdf["started_at"] = gdf["started_at"].apply(lambda x: pd.to_datetime(x))
gdf["finished_at"] = gdf["finished_at"].apply(lambda x: pd.to_datetime(x))
gdf["time_delta"] = gdf["finished_at"] - gdf["started_at"]

# %%
gdf.head()

# %% [markdown]
# # Trips
# 

# %%
"count_trips", len(gdf)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize_medium)

ax0 = sns.barplot(data=gdf["mode"].value_counts(), ax=axes[0])
ax0.bar_label(ax0.containers[0],fontsize=fontsize_medium)

ax1 = sns.lineplot(data=gdf["length"].value_counts(), ax=axes[1])

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.1)
axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.1)

axes[0].set_xlabel("Mode of Transportation", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Trip Length", fontsize=fontsize_medium)
#axes[1].set_ylabel("")


plt.tight_layout()



# %%
# count trips, length, duration

# %%
user_df = gdf[["user_id", "gender", "age", "education", 'activity_work', 'activity_leisure', "activity_errand"]].drop_duplicates(subset=['user_id'])
len(user_df)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=figsize_medium)
education_order = ["PhD", "Bachelor/Master/Diploma", "High School" , "Professional training", "Secondary School", "Middle School","no degree"]

ax0 = user_df["gender"].value_counts().plot(kind="bar", xlabel="", title="gender", fontsize=fontsize_medium, ax=axes[0])
ax0.bar_label(ax0.containers[0],fontsize=fontsize_medium)
axes[0].title.set_size(fontsize_medium)

ax1 = user_df["age"].plot(kind="hist",title="age", xlabel="years", ylabel="", fontsize=fontsize_medium,ax=axes[1])
ax1 =ax1.bar_label(ax1.containers[0],fontsize=fontsize_medium)
axes[1].title.set_size(fontsize_medium)

ax2 = user_df["education"].value_counts().reindex(education_order[::-1]).plot(kind="bar", title="education", xlabel="",fontsize=fontsize_medium,ax=axes[2])
ax2 =ax2.bar_label(ax2.containers[0],fontsize=fontsize_medium)
axes[2].title.set_size(fontsize_medium)

axes[0].set_xlabel("Gender", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Age", fontsize=fontsize_medium)
axes[1].set_ylabel("")
axes[2].set_xlabel("Education", fontsize=fontsize_medium)
axes[2].set_ylabel("")

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)
axes[2].set_title("", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)
axes[2].tick_params(axis='x', labelsize=fontsize_medium)
axes[2].tick_params(axis='y', labelsize=fontsize_medium)

axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.1)
axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.1)
axes[2].set_ylim(0,axes[2].get_ylim()[1]*1.1)

axes[2].set_xticklabels(["No Deg.", "Middle Sch.", "Secondary Sch.", "Prof. Training", "High Sch.", "Academic Deg.", "PhD" ])
plt.tight_layout()

#fig.savefig('../plots/evaluation/person_summary.png', dpi=dpi)

# %%
df0 =user_df["activity_work"].value_counts().reset_index().assign(activity ="work").rename(columns={"activity_work": "frequency"})
df1 =user_df["activity_leisure"].value_counts().reset_index().assign(activity ="leisure").rename(columns={"activity_leisure": "frequency"})
df2 =user_df["activity_errand"].value_counts().reset_index().assign(activity ="errand").rename(columns={"activity_errand": "frequency"})
original_wle_df2 = pd.concat([df0,df1,df2])

df0 =processed_wle_df["activity_work"].value_counts().reset_index().assign(activity ="work").rename(columns={"activity_work": "frequency"})
df1 =processed_wle_df["activity_leisure"].value_counts().reset_index().assign(activity ="leisure").rename(columns={"activity_leisure": "frequency"})
df2 =processed_wle_df["activity_errand"].value_counts().reset_index().assign(activity ="errand").rename(columns={"activity_errand": "frequency"})
processed_wle_df2 = pd.concat([df0,df1,df2])

# %%
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(figsize_medium))
activity_order = ["more than 5 times a week", "5 times a week","4 times a week","3 times a week","Twice a week","Once a week","Never"][::-1]
activits = ["Work", "Leisure", "Errand"]
ax0 = sns.barplot(data=original_wle_df2,x = "frequency", y="count", hue="activity", order=activity_order,ax=axes[0])

ax1 = sns.barplot(data=processed_wle_df2,x = "frequency", y="count", hue="activity", order=activity_order,ax=axes[1])

ax0.tick_params(axis='x', rotation=90, labelsize=fontsize_medium)
ax1.tick_params(axis='x', rotation=90, labelsize=fontsize_medium)

ax0.tick_params(axis='y',  labelsize=fontsize_medium)
ax1.tick_params(axis='y',  labelsize=fontsize_medium)


ax0.legend(fontsize=fontsize_medium)
ax1.legend(fontsize=fontsize_medium)

xticklabels = ["0x/week", "1x/week","2x/week","3x/week","4x/week","5x/week" ,  ">5x/week"]
ax0.set_xticklabels(xticklabels)
ax1.set_xticklabels(xticklabels)

ax0.set_xlabel("Original", fontsize=fontsize_medium)
ax0.set_ylabel("Count", fontsize=fontsize_medium)
ax1.set_xlabel("Processed", fontsize=fontsize_medium)
ax1.set_ylabel("")

plt.tight_layout()

fig.savefig('../plots/evaluation/wle_compare_bar.png', dpi=dpi)

# %%
'''
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(figsize_medium[0], figsize_medium[1]*2.5))
activity_order = ["more than 5 times a week", "5 times a week","4 times a week","3 times a week","Twice a week","Once a week","Never"][::-1]
activits = ["Work", "Leisure", "Errand"]

for i,activity in enumerate(activits):
    axes[i,0] = sns.barplot(data=original_wle_df2[original_wle_df2["activity"] == activity.lower()],x = "frequency", y="count", order=activity_order,ax=axes[i,0])
    
    for j in axes[i,0].containers:
        axes[i,0].bar_label(j,fontsize=fontsize_medium)


    axes[i,1] = sns.barplot(data=processed_wle_df2[processed_wle_df2["activity"] == activity.lower()],x = "frequency", y="count", order=activity_order, ax=axes[i,1])
    
    for j in axes[i,1].containers:
        axes[i,1].bar_label(j,fontsize=fontsize_medium)

    axes[i,0].tick_params(axis='x', rotation=90, labelsize=fontsize_medium)
    axes[i,0].set_ylabel("Count", fontsize=fontsize_medium)
    axes[i,0].set_xlabel("Frequency", fontsize=fontsize_medium)
    axes[i,0].set_title("", fontsize=fontsize_medium)
        
    axes[i,1].tick_params(axis='x', rotation=90, labelsize=fontsize_medium)
    
    axes[i,0].set_xlabel(f"{activity} Original", fontsize=fontsize_medium)
    axes[i,1].set_xlabel(f"{activity} Processed", fontsize=fontsize_medium)


    axes[i,0].set_title(f"", fontsize=fontsize_medium)
    axes[i,1].set_title(f"", fontsize=fontsize_medium)

    axes[i,0].tick_params(axis='x', labelsize=fontsize_medium)
    axes[i,1].tick_params(axis='x', labelsize=fontsize_medium)
    axes[i,0].tick_params(axis='y', labelsize=fontsize_medium)
    axes[i,1].tick_params(axis='y', labelsize=fontsize_medium)

    axes[i,0].set_ylim(0,axes[i,0].get_ylim()[1]*1.1)
    axes[i,1].set_ylim(0,axes[i,1].get_ylim()[1]*1.1)

    if i != 2:
        axes[i,0].set_xticks("")
        axes[i,1].set_xticks("")
        axes[i,1].set_ylabel("", fontsize=fontsize_medium)

axes[2,1].set_ylabel("", fontsize=fontsize_medium)
xticklabels = ["0x/week", "1x/week","2x/week","3x/week","4x/week","5x/week" ,  ">5x/week"]
axes[2,0].set_xticklabels(xticklabels)
axes[2,1].set_xticklabels(xticklabels)
  


plt.tight_layout()

#fig.savefig('../plots/evaluation/wle_compare_bar.png', dpi=dpi)
'''

# %%
'''
activity_order = ["more than 5 times a week", "5 times a week","4 times a week","3 times a week","Twice a week","Once a week","Never"][::-1]

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize_large)

ax1 = user_df["activity_work"].value_counts().reindex(activity_order).plot(kind="bar",xlabel="", title="work original",fontsize=fontsize_medium,ax=axes[0])
ax1 =ax1.bar_label(ax1.containers[0],fontsize=fontsize_medium)
axes[0].title.set_size(fontsize_medium)

ax2 = processed_wle_df["activity_work"].value_counts().reindex(activity_order).plot(kind="bar", xlabel="", title="work processed",fontsize=fontsize_medium,ax=axes[1])
ax2 =ax2.bar_label(ax2.containers[0],fontsize=fontsize_medium)
axes[1].title.set_size(fontsize_medium)

plt.tight_layout()

fig.savefig('../plots/results/work_compare_bar.png')


fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = user_df["activity_leisure"].value_counts().reindex(activity_order).plot(kind="bar", title="leisure original", xlabel="",ax=axes[0])
ax1 =ax1.bar_label(ax1.containers[0])

ax2 = processed_wle_df["activity_leisure"].value_counts().reindex(activity_order).plot(kind="bar", title="leisure processed", xlabel="",ax=axes[1])
ax2 =ax2.bar_label(ax2.containers[0])

fig, axes = plt.subplots(nrows=1, ncols=2)
ax1 = user_df["activity_errand"].value_counts().reindex(activity_order).plot(kind="bar", title="errand original", xlabel="",ax=axes[0])
ax1 =ax1.bar_label(ax1.containers[0])

ax2 = processed_wle_df["activity_errand"].value_counts().reindex(activity_order).plot(kind="bar", title="errand processed",xlabel="",ax=axes[1])
ax2 =ax2.bar_label(ax2.containers[0])
'''
""

# %%
processed_wle_df["count_tracks"] = 6
user_df["count_tracks"] = [len(gdf[gdf["user_id"] == user_id]) for user_id in user_df["user_id"]]

# %%
'''
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

ax0 = user_df["count_tracks"].plot(kind="hist", bins =[0,100,200,300,400,500,600,700,800,900], ax=axes[0])
ax0.bar_label(ax0.containers[0], fontsize=fontsize_medium)

ax1 = processed_wle_df["count_tracks"].plot(kind="hist", bins =[0,100,200,300,400,500,600,700,800,900], ax=axes[1])
ax1.bar_label(ax1.containers[0], fontsize=fontsize_medium)

axes[0].set_xlabel("Trips Per Person", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Trips Per Person", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("Trips Per Person Original", fontsize=fontsize_medium)
axes[1].set_title("Trips Per Person Processed", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

plt.tight_layout()

#fig.savefig('../plots/evaluation/tripcountpp_compare_bar.png')
'''

# %%
apperance_counter = Counter(list(sample_to_user_id.values()))
apperance_user_ids =  apperance_counter.keys()
apperance_counts = apperance_counter.values()
apperances_processed_df = pd.DataFrame({"user": apperance_user_ids, "apperances": apperance_counts})
apperances_original_df = pd.DataFrame({"user": apperance_user_ids, "apperances": 1})

apperances_processed_df.head()

# %%
'''
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

bins = [10*i for i in range(16)]
ax0 = apperances_original_df["apperances"].plot(kind="hist", bins=bins, ax=axes[0])
ax0.bar_label(ax0.containers[0], fontsize=fontsize_medium)

ax1 = apperances_processed_df["apperances"].plot(kind="hist", bins=bins, ax=axes[1])
ax1.bar_label(ax1.containers[0], fontsize=fontsize_medium)

#sns.histplot(data=apperances_original_df, x="apperances",bins =bins, ax=axes[0])
#sns.histplot(data=apperances_processed_df, x="apperances", bins =bins, ax=axes[1])


axes[0].set_xlabel("Apperances Per Person", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Apperances Per Person", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("Apperances Original", fontsize=fontsize_medium)
axes[1].set_title("Apperances Processed", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)

axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

plt.tight_layout()

#fig.savefig('../plots/evaluation/apperancespp_compare_bar.png')
'''

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )
bins = [0,100,200,300,400,500,600,700,800,900]
ax0 = user_df["count_tracks"].plot(kind="hist", ax=axes[0])
ax0.bar_label(ax0.containers[0], fontsize=fontsize_medium)


axes[0].set_xlabel("Trips Per Person Original", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[0].set_title("", fontsize=fontsize_medium)
axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)

############

ax1 = apperances_processed_df["apperances"].plot(kind="hist", ax=axes[1])
ax1.bar_label(ax1.containers[0], fontsize=fontsize_medium)

axes[1].set_xlabel("Apperances Per Person Processed", fontsize=fontsize_medium)
axes[1].set_ylabel("")
axes[1].set_title("", fontsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)


axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[1].set_xlim(0,axes[1].get_xlim()[1])

axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.1)
axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.1)

plt.tight_layout()

#fig.savefig('../plots/evaluation/tripcountpp_compare_bar.png', dpi = dpi)

# %%
user_df["count_tracks"].sum()

# %%
len(user_df)

# %%
# 4,5,6,7,8
# Personal min 6 trips
trip_counts = [4,5,6,7,8]
user_ge4trips_df = user_df[user_df["count_tracks"] >= 4]
user_ge5trips_df = user_df[user_df["count_tracks"] >= 5]
user_ge6trips_df = user_df[user_df["count_tracks"] >= 6]
user_ge7trips_df = user_df[user_df["count_tracks"] >= 7]
user_ge8trips_df = user_df[user_df["count_tracks"] >= 8]

user_trips_dfs = [user_ge4trips_df, user_ge5trips_df, user_ge6trips_df ,user_ge7trips_df, user_ge8trips_df]

[len(user_trips_df)for user_trips_df in user_trips_dfs]

# %% [markdown]
# # Datasets

# %%
times_min1 = [3,4,5,6,7]
resolutions = [7,8,9]

'''
for resolution in resolutions:
    for t in times_min1:
        dataset_name = f"dataset_res{resolution}"
        loader = TrajectoryDatasetLoader(f"./data/{dataset_name}_v2.json", 3, 3)
        dataset = loader.get_dataset()

        #snapshot = next(iter(dataset))
        res = resolution
        timesteps = t+1
        snapshots = dataset.snapshot_count
        
        dataset_summary_dict["resolution"].append(resolution)
        dataset_summary_dict["timesteps"].append(t+1)
        dataset_summary_dict["features"].append(7)
        dataset_summary_dict["snapshots"].append(sh_cnt)
        dataset_summary_dict["nodes"].append(snapshot.x.shape[0])
        dataset_summary_dict["edges"].append(snapshot.edge_index.shape[1])
        print(dataset_summary_dict)
        
        print("resolution: ",resolution,"timesteps: ", timesteps,"snapshots: ", snapshots)
'''
dataset_summary_dict = dict()
dataset_summary_dict["nodes"] = [89, 89, 89, 89, 89, 511, 511, 511, 511, 511, 3265,3265,3265,3265,3265]
dataset_summary_dict["edges"] = [456, 456, 456, 456, 456, 2844, 2844, 2844, 2844, 2844, 18940,18940,18940,18940,18940]
dataset_summary_dict["snapshots"] = [8498, 6765, 5595, 4771, 4153, 8498, 6765, 5595, 4771, 4153,8498, 6765, 5595, 4771, 4153]
dataset_summary_dict["timesteps"] = [4, 5, 6, 7, 8,4, 5, 6, 7, 8,4, 5, 6, 7, 8,]
dataset_summary_dict["features"] = [7, 7, 7, 7, 7,7, 7, 7, 7, 7,7, 7, 7, 7, 7] 
dataset_summary_dict["resolution"]  = [7, 7, 7, 7, 7,8,8,8,8,8,9,9,9,9,9]

# %%
regions_enhanced_res7 = pd.read_pickle("../data/regions_enhanced_res7.pkl")
regions_enhanced_res8 = pd.read_pickle("../data/regions_enhanced_res8_v2.pkl")

# %%
def build_graph_no_features(node_features, edge_list):
    graph = nx.Graph()
    for index, node in node_features.iterrows():
        graph.add_node(node.name)
    
    for u,v in edge_list:
        graph.add_edge(u, v)
    return graph

# %%
# Graphs
edge_list7, _ = build_edges(regions_enhanced_res7)
graph7 = build_graph(regions_enhanced_res7,edge_list7)

edge_list8, _ = build_edges(regions_enhanced_res8)
graph8 = build_graph(regions_enhanced_res8,edge_list8)

edge_list9, _ = build_edges(regions9)
graph9 = build_graph_no_features(regions9,edge_list9)

graph7, graph8, graph9

# %%
pos7=nx.kamada_kawai_layout(graph7)
pos8=nx.kamada_kawai_layout(graph8)
pos9=nx.kamada_kawai_layout(graph9)

# %%
fig, axes = plt.subplots(nrows=1, ncols=3,figsize=figsize_medium )

ax0 = nx.draw(graph7,pos7, cmap = plt.get_cmap('jet'),  node_size = 10, ax=axes[0],width=1)
ax1 = nx.draw(graph8,pos8, cmap = plt.get_cmap('jet'), node_size = 1, ax=axes[1],width=0.1)
ax2 = nx.draw(graph9,pos9, cmap = plt.get_cmap('jet'),node_size = .1,ax=axes[2],width=0.1)

axes[0].set_title("Region Graph 7",fontsize=fontsize_medium)
axes[1].set_title("Region Graph 8",fontsize=fontsize_medium)
axes[2].set_title("Region Graph 9",fontsize=fontsize_medium)

plt.tight_layout()
plt.show()

fig.savefig('../plots/evaluation/basegraphs_res789.png', dpi=dpi)

# %% [markdown]
# # Mobility

# %%
# TODO!!!

WGS84_EPSG = "EPSG:4326"
METRIC_EPSG = "EPSG:31468"

# %%
#user_df[user_df["count_tracks"]==6]["user_id"]
#test_user_id = "a346dc45-399c-4b6a-86f2-a6740bb25ada" # okay
#test_user_id = "66ee1d65-7643-49cd-be50-8f39c0cf6fc1"  # good
#test_user_id = "b7927773-4781-4ce7-8640-83ccb571dced"

test_user_id = "174d627f-33a5-4d75-bdd8-0045b8a16c46" # best

# %%
test_user_gdf = gdf[gdf["user_id"]==test_user_id ]

# hexagons
hexes = test_user_gdf["h3_hex"]

# plot lines
geoms = test_user_gdf["geom"].to_crs(WGS84_EPSG)
folium_map = plot_regions(region_muc, colormap=["rgba(0,0,0,0.1)"])
for ls in geoms.values:
    yy, xx  = list(ls.coords.xy)
    cords_tupels = tuple(zip(xx.tolist(),yy.tolist()))
    my_PolyLine=folium.PolyLine(locations=cords_tupels)
    folium_map.add_child(my_PolyLine)

folium_map

# %%
folium_map = plot_regions(region_muc, colormap=["rgba(0,0,0,0)"])

for hexe in hexes:
    for h in hexe:
        bounds = h3.cell_to_boundary(h)
        bounds_corrected = [(y,x) for x,y in bounds]
        my_polygon = sp.Polygon(bounds_corrected)
        my_polygon = folium.Choropleth(geo_data=my_polygon,style_function={'fillColor': 'blue', "opacity": 1000})
        
        folium_map.add_child(my_polygon)


folium_map

# %%
# Plot 1 Trip, Its encoding
# Plot all trips and their encoding
# processed_wle_tensor

# %%
## Mobility KPIs pure data

users = gdf["user_id"].unique()
original_rogs = []
original_krogs = []

original_jmpss = []
original_njmpss = []

for i,user in enumerate(users):
    
    user_dict = dict()
    entries = gdf[gdf["user_id"] == user]

    # jump sizes
    startps = list(entries["start"].values)
    endps = list(entries["end"].values)
    jump_sizes =[startp.distance(endp) for startp, endp in zip(startps, endps)]
    original_jmpss.append(jump_sizes)

    # radius of gyration
    points = startps+ endps
    centroid = MultiPoint(points).centroid
    k = len(points)
    rg = np.sqrt(1/k * np.sum([centroid.distance(p)**2 for p in points]))
    original_rogs.append(rg)

    # normalized jump sizes
    normalized_jump_sizes = [ js/rg for js in jump_sizes]
    original_njmpss.append(normalized_jump_sizes)

    # top k radius of gyration based on hex membership (for res 8)
    hexess = []
    for hexs in entries["h3_hex"].values:
        hexess.append(hexs[0])
        hexess.append(hexs[-1])
    
    top_k_hexess = top_k_frequent_elements(hexess, 2)
    transformer = Transformer.from_crs(WGS84_EPSG, METRIC_EPSG)    
    top_k_source_dest_points_t = ([Point(transformer.transform(*h3.cell_to_latlng(hex))) for hex in top_k_hexess])
    centroid_k = MultiPoint(top_k_source_dest_points_t).centroid
    k = len(top_k_source_dest_points_t)    
    krg = np.sqrt(1/k * np.sum([centroid_k.distance(p)**2 for p in top_k_source_dest_points_t]))
    original_krogs.append(krg)

# %%
original_jmpss_flat = list(flatten(original_jmpss))
original_njmpss_flat = list(flatten(original_njmpss))

# %%
user_snapshot_dict = dict()

for i,my_snapshot in enumerate(processed_all_trips_tensor):
    uid = sample_to_user_id[i]
    
    if uid not in user_snapshot_dict.keys():
        user_snapshot_dict[uid] = []
    user_snapshot_dict[uid].append(my_snapshot)

# %%
processed_rogs_dict = {uid : radius_of_gyration(trip_histories,ids_to_hex_names, distances_df) for uid, trip_histories in user_snapshot_dict.items()}
processed_krogs_dict = {uid: k_radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_snapshot_dict.items()}

# %%
processed_jmpss_ps = [get_jump_sizes(trip_histories,ids_to_hex_names, distances_df) for trip_histories in processed_all_trips_tensor]
processed_jmpss = list(flatten(processed_jmpss_ps))

processed_njmpss = []
for sample_id, uid in sample_to_user_id.items():
    for js in processed_jmpss_ps[sample_id]:
        processed_njmpss.append(js/processed_rogs_dict[uid])

# %%
jmps_proccesed_df = pd.DataFrame({"jmps": processed_jmpss, "njmps": processed_njmpss})
jmps_original_df = pd.DataFrame({"jmps": original_jmpss_flat, "njmps":original_njmpss_flat})

jmps_df = pd.concat([jmps_proccesed_df.assign(source="Processed"), jmps_original_df.assign(source="Original")])

# %%
rogs_processed_df = pd.DataFrame({"rog": list(processed_rogs_dict.values()), "krog": list(processed_krogs_dict.values())})
rogs_original_df = pd.DataFrame({"rog": original_rogs, "krog": original_krogs})

rogs_df = pd.concat([rogs_processed_df.assign(source="Processed"), rogs_original_df.assign(source="Original")])

# %%
# rog
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )
ax0 = sns.histplot(data=rogs_df, x="rog", hue="source", ax=axes[0], multiple="dodge")
ax1 = sns.histplot(data=rogs_df, x="krog", hue="source", ax=axes[1] , multiple="dodge")

axes[0].set_xlabel("Radius of Gyration [m]", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("K-Radius of Gyration [m]", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

ax0.legend(["Original", "Processed"], fontsize=fontsize_medium)
ax1.legend(["Original", "Processed"], fontsize=fontsize_medium)

axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[1].set_xlim(0,axes[1].get_xlim()[1])

plt.tight_layout()

#fig.savefig('../plots/evaluation/rog_compare_orig_vs_proces.png', dpi = dpi)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium )
ax = sns.histplot(data=jmps_df, x="jmps", hue="source", ax=ax,multiple="dodge")

ax.set_xlabel("Jump Size [m]", fontsize=fontsize_medium)
ax.set_ylabel("Count", fontsize=fontsize_medium)
ax.set_title("", fontsize=fontsize_medium)
ax.tick_params(axis='x', labelsize=fontsize_medium)
ax.tick_params(axis='y', labelsize=fontsize_medium)

ax.legend(["Original", "Processed"], fontsize=fontsize_medium)
ax.set_xlim(0,ax.get_xlim()[1])

plt.tight_layout()

fig.savefig('../plots/evaluation/jump_sizes_compare.png', dpi=dpi)

# %%
'''
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )
ax0 = sns.histplot(data=jmps_df, x="jmps", hue="source", ax=axes[0],multiple="dodge")
ax1 = sns.histplot(data=jmps_df, x="njmps",hue="source", ax=axes[1],multiple="dodge")

axes[0].set_xlabel("Jump Size [m]", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Normalized Jump Size ", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

ax0.legend(["Original", "Processed"], fontsize=fontsize_medium)
ax1.legend(["Original", "Processed"], fontsize=fontsize_medium)

axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[1].set_xlim(0,axes[1].get_xlim()[1])

plt.tight_layout()

fig.savefig('../plots/evaluation/krog_compare_bar.png')
'''

# %%
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium )

ax = sns.scatterplot(data=rogs_df, x="rog",y="krog", hue="source", ax=ax, s= 15)


ax.set_xlabel("Radius of Gyration [m]", fontsize=fontsize_medium)
ax.set_ylabel("K-Radius of Gyration [m]", fontsize=fontsize_medium)

ax.set_title("", fontsize=fontsize_medium)

ax.tick_params(axis='x', labelsize=fontsize_medium)
ax.tick_params(axis='y', labelsize=fontsize_medium)

ax.legend(fontsize=fontsize_medium)

ax.set_xlim(0,ax.get_xlim()[1])
ax.set_ylim(0,ax.get_ylim()[1])

plt.tight_layout()

#fig.savefig('../plots/evaluation/rogvskrog_compare_scat.png', dpi=dpi)

# %%
print(jmps_df.groupby("source").describe().T.to_latex())

# %%
print(rogs_df.groupby("source").describe().T.to_latex())


