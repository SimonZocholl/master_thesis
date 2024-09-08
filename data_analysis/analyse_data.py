# %%
import sys
sys.path.insert(0, '/workspaces/trackgenerator')

import folium
import pandas as pd
import geopandas as gpd
import h3 as h3

import seaborn as sns
import matplotlib.pyplot as plt

from shapely.geometry import LineString, Point

from srai.plotting import plot_regions
from srai.regionalizers.geocode import geocode_to_region_gdf
from srai.h3 import h3_to_geoseries
from h3ronpy.arrow.vector import ContainmentMode, wkb_to_cells
from master_thesis.utils.utils import string_to_lineString, linestring_to_polyline, my_shapely_geometry_to_h3

#import difftrackgenerator.config as config

%load_ext autoreload
%autoreload 2
# add final destination
# add poi for final destination
# plott poi

# %% [markdown]
# # Questions
# 
# * ids
#   * How many trajectories? 47282
#   * How many unique persons? 894
# 
# * time
#   * start, end of study: 25 May, 2022 04 June, 2023
#   * study time delta: 375 days 03:25:14
#   * trajectory time: min 00:00:05, max 00:18:13, mean 00:13:12, mead 00:10:44
# 
# * length: min 35, max 45778, mean 3815, mead 3815
# 
# * confirmed:  True 37117, False 10165
# 
# * merged: False 46969, True 313
# 
# * updated_at start, end:  25 May, 2022 14 June, 2023
# 
# * Trajectories
#   * contains: 130 cases of MULTILINESTRING -> remove
#   * visualize a single track
#   * visualize all tracks?
#   * visualize all tracks form a single person
#   * put single, all from a person, all tracks in h3 grids
# 

# %%
WGS84_EPSG = "EPSG:4326"
METRIC_EPSG = "EPSG:31468"
resolution = 8

# %%


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

# remove multi line strings (130 occurances)
df = df[~df['geom'].str.contains("MULTILINESTRING")]

# parse time information
# https://jeffkayser.com/projects/date-format-string-composer/index.html
df["started_at"] = pd.to_datetime(df["started_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)
df["finished_at"] = pd.to_datetime(df["finished_at"], format="%Y-%m-%d %H:%M:%S%z",utc=True)
#df["updated_at"] = pd.to_datetime(df["updated_at"], format="%Y-%m-%d %H:%M:%S.%f")

# parse lineStrings
df["geom"] = df["geom"].apply(string_to_lineString)

df["age"] = df["age"].astype(int)

survery_columns = ['user_id','gender', 'age', 'income_net', 'job', 'education', 'hh_size', 'hh_children', 'usage_car', 'activity_work', 'activity_leisure', 'car', 'activity_errand', 'distance_work', 'distance_leisure', 'distance_errand']
survey_df =  df[survery_columns].drop_duplicates()

area = geocode_to_region_gdf("Munich, Germany")
gdf = gpd.GeoDataFrame(df, crs=WGS84_EPSG,geometry="geom")
gdf["polyline"] = gdf["geom"].apply(linestring_to_polyline)
gdf["h3_hex"] = gdf["geom"].apply(lambda x: my_shapely_geometry_to_h3(x))
gdf["start"] = gdf["geom"].apply(lambda x: Point(x.coords[0]))
gdf["end"] = gdf["geom"].apply(lambda x: Point(x.coords[-1]))
gdf["start_h3_hex"] = gdf["start"].apply(lambda x: my_shapely_geometry_to_h3(x))
gdf["end_h3_hex"] = gdf["end"].apply(lambda x: my_shapely_geometry_to_h3(x))

gdf.head()

# %% [markdown]
# # Simple data analysis

# %%
# unique values
print("number of ids: ", len(gdf["id"].unique())) # 47282
print("number of user_id: ", len(gdf["user_id"].unique()) )# 894

# %%
user_ids = gdf["user_id"]
user_df = pd.DataFrame(gdf["user_id"])
survey_df["count_tracks"] = [len(gdf[gdf["user_id"] == user_id]) for user_id in survey_df["user_id"]]

# %%
print(survey_df["count_tracks"].describe())
sns.histplot(data=survey_df, x="count_tracks")

# %%
time_df = pd.DataFrame({"time_delta": gdf["finished_at"] - gdf["started_at"]}) 
histplot = sns.histplot(time_df['time_delta']/pd.Timedelta(minutes=1))
print(time_df["time_delta"].describe())
histplot.set_xlim(left=0, right=100)

# %%
sns.histplot(gdf["started_at"])

# %%
print("started_at start, end: ", gdf["started_at"].min().strftime("%d %B, %Y"), gdf["started_at"].max().strftime("%d %B, %Y"))
print("finished_at start, end: ", gdf["finished_at"].min().strftime("%d %B, %Y"), gdf["finished_at"].max().strftime("%d %B, %Y"))
print("time range: ", gdf["started_at"].max()-gdf["started_at"].min())

# %%
print(gdf["length"].describe())
sns.histplot(gdf["length"])

# %%
print(gdf["confirmed"].value_counts())
sns.barplot(gdf["confirmed"].value_counts())

# %% [markdown]
# # User analysis

# %%
survey_df.columns

# %%
print(survey_df["gender"].value_counts())
sns.barplot(survey_df["gender"].value_counts())

# %%
print(survey_df["age"].describe())
sns.histplot(survey_df["age"])

# %%
print(survey_df["income_net"].value_counts())
sns.barplot(survey_df["income_net"].value_counts())

# %%
print(survey_df["job"].value_counts())
sns.barplot(survey_df["job"].value_counts())

# %%
print(survey_df["education"].value_counts())
education_order = ["PhD", "Bachelor/Master/Diploma", "High School" , "Professional training", "Secondary School", "Middle School","no degree"]
ax = sns.barplot(survey_df["education"].value_counts(), order=education_order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# %%
print(survey_df["hh_size"].value_counts())
sns.barplot(survey_df["hh_size"].value_counts())

# %%
print(survey_df["hh_children"].value_counts())
sns.barplot(survey_df["hh_children"].value_counts())

# %%
print(survey_df["usage_car"].value_counts())
ax = sns.barplot(survey_df["usage_car"].value_counts())
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# %%
print(survey_df["activity_leisure"].value_counts())
activity_leisure_order = ["more than 5 times a week", "5 times a week","4 times a week","3 times a week","Twice a week","Once a week","Never"]
ax = sns.barplot(survey_df["activity_leisure"].value_counts(), order=activity_leisure_order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# %%
print(survey_df["activity_work"].value_counts())
activity_work_order = activity_leisure_order
ax = sns.barplot(survey_df["activity_work"].value_counts(), order=activity_leisure_order)
ax.set_xticklabels(ax.get_xticklabels(), rotation=40, ha="right")
plt.tight_layout()
plt.show()

# %%
print(survey_df["car"].value_counts())
sns.barplot(survey_df["car"].value_counts())

# %% [markdown]
# # Trajectory Analysis

# %%
# WGS84_EPSG = "EPSG:4326"
# METRIC_EPSG = "EPSG:31468"

gdf.head()

# %%
# given Length == calculated length
geom_len = gdf["geom"].to_crs(crs=METRIC_EPSG).length
given_len = df["length"]

len_df = pd.DataFrame({"geom_len": geom_len, "given_len": given_len, "delta_len": (geom_len- given_len)})

print(len_df["delta_len"].describe())
histplot = sns.histplot(len_df["delta_len"])
histplot.set_xlim(-50,50)

# %%
plot_id = 'fd48b3b3-6c9e-448e-a169-2643aa1b7aa2'
folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
polyline = gdf[gdf["id"]== plot_id]["polyline"][0]
h3_hexs = gdf[gdf["id"]== plot_id]["h3_hex"][0]
for h3_hex in h3_hexs:
    sim_geo = h3_to_geoseries(h3_hexs)
    geo_j = sim_geo.to_json()
    geo_j = folium.GeoJson(data=geo_j, style_function= lambda x: {"opacity": 0.1, "color": "green"})
    folium_map.add_child(geo_j)

folium_map.add_child(polyline)
folium_map

# %%
plt_user_id = '8fd512c1-82fe-4af0-b1bc-298389bdf3b9'
folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
polylines = gdf[gdf["user_id"]== plt_user_id]["polyline"]

h3_hexs_list = list(gdf[gdf["user_id"]== plt_user_id]["h3_hex"])

for i, h3_hexs in enumerate(h3_hexs_list):
    for j,h3_hex in enumerate(h3_hexs):
        sim_geo = h3_to_geoseries(h3_hexs)
        geo_j = sim_geo.to_json()
        geo_j = folium.GeoJson(data=geo_j, style_function= lambda x: {"opacity": 0.1, "color": "green"})
        folium.Tooltip(f"Hex: {i,j}").add_to(geo_j)
        folium_map.add_child(geo_j)


for i,polyline in enumerate(polylines):
    folium.Tooltip(f"Traj: {i}").add_to(polyline)
    folium_map.add_child(polyline)

folium_map

# %%
folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")
polylines = gdf["polyline"]

#for i,polyline in enumerate(polylines):
#    folium.Tooltip(f"Traj: {i}").add_to(polyline)
#    folium_map.add_child(polyline)

#folium_map
# 1234

# %% [markdown]
# # POI Analysis

# %%
# make non intersecting ???

work_set = {'agency', 'archaeological', 'art', 'arts', 'atm', 'bank', 'bookmaker', 'building','cannabis', 'carpet', 'casino', 'centre', 'clinic', 'college', 'commercial', 'computer', 'construction', 'consulate', 'copyshop', 'courthouse', 'craft', 'dentist', 'detached', 'diplomatic', 'directors', 'diving', 'doctors', 'driving', 'electrical', 'electronics', 'facility', 'farm', 'fire', 'furniture', 'gallery', 'government', 'hairdresser', 'health', 'hospital', 'hotel', 'industrial', 'kindergarten', 'kitchen', 'laundry', 'library', 'machine', 'mall', 'marketplace', 'medical', 'mobile', 'model', 'museum', 'office', 'outpost', 'pawnbroker', 'pharmacy', 'power', 'prison', 'public', 'radiotechnics', 'rental', 'repair', 'residential', 'restaurant', 'retail', 'riding', 'school', 'scuba', 'security', 'semidetached', 'service', 'services', 'shelter', 'shop', 'station', 'storage', 'supermarket', 'swimming', 'synagogue', 'tailor', 'taxi', 'trailer', 'train', 'transportation', 'university', 'veterinary', 'video', 'warehouse', 'works', 'worship', 'auxiliary', 'barn', 'books', 'castle', 'conservatory', 'court', 'de', 'garage', 'garages', 'motorcycle', 'nursing', 'artwork', 'baby', 'bag', 'bakery', 'bar', 'bath', 'books', 'bowling', 'bungalow', 'castle', 'conservatory', 'court', 'de', 'dog', 'fashion', 'garage', 'garages', 'garden', 'motorcycle', 'nursing', 'rink', 'trophy'}
leisure_set = {'alley', 'area', 'arts', 'atm', 'attraction', 'bbq', 'beach', 'biergarten', 'bookmaker', 'bridge', 'cafe', 'camp', 'casino', 'centre', 'chapel', 'cinema', 'city', 'civic', 'coffee', 'collector', 'community', 'confectionery', 'courthouse', 'craft', 'cream', 'fountain', 'gallery', 'games', 'golf', 'guest', 'historic', 'hostel', 'leisure', 'mall', 'marketplace', 'massage', 'miniature', 'monastery', 'monument', 'mosque', 'motel', 'museum', 'music', 'musical', 'nature', 'newsagent', 'nightclub', 'park', 'picnic', 'pitch', 'place', 'playground', 'pool', 'recreation', 'restaurant', 'roof', 'sauna', 'scuba', 'seating', 'sports', 'stadium', 'store', 'swimming', 'theatre', 'theme', 'tourism', 'track', 'travel', 'viewpoint', 'works', 'zoo'}
errand_set = {'accessories', 'aids', 'alcohol', 'amenity', 'antiques', 'apartment', 'apartments', 'appliance', 'bathroom', 'beauty', 'bed', 'beverages', 'bicycle', 'blind', 'boat', 'bookcase', 'boutique', 'box', 'bunker', 'bureau', 'bus', 'butcher', 'cabin', 'camera', 'candles', 'car', 'caravan', 'carport', 'change', 'charging', 'charity', 'cheese', 'chemist', 'childcare', 'chocolate', 'church', 'cleaner', 'cleaning', 'clothes', 'collector', 'container', 'convenience', 'cosmetics', 'course', 'cowshed', 'cream', 'curtain', 'dairy', 'decoration', 'deli', 'department', 'disposal', 'doityourself', 'dormitory', 'drinking', 'dry', 'e-cigarette', 'entrance', 'erotic', 'fabric', 'fast', 'fire', 'fireplace', 'fishing', 'fitness', 'florist', 'food', 'frame', 'fuel', 'funeral', 'furnishing', 'furniture', 'gate', 'general', 'gift', 'glaziery', 'goods', 'grandstand', 'grave', 'greengrocer', 'greenhouse', 'grooming', 'ground', 'hairdresser', 'hall', 'hand', 'hangar', 'hardware', 'hearing', 'herbalist', 'hifi', 'highway', 'home', 'house', 'houseware', 'hunting', 'hut', 'ice', 'information', 'inspection', 'instrument', 'interior', 'internet', 'jewelry', 'kiosk', 'landuse', 'leather', 'library', 'lighting', 'locksmith', 'lottery', 'made', 'man', 'manor', 'medical', 'memorial', 'mobile', 'natural', 'nutrition', 'of', 'optician', 'outdoor', 'paint', 'parking', 'parts', 'party', 'pasta', 'pastry', 'pavilion','perfumery', 'pet', 'phone', 'photo', 'place', 'plant', 'police', 'post', 'presbytery', 'pub', 'pyrotechnics', 'recreation', 'recycling', 'religious', 'reserve', 'rest', 'retail', 'roof', 'sand', 'school', 'seafood', 'second', 'services', 'sewing', 'shed', 'shelter', 'shoes', 'shop', 'shower', 'shrine', 'site', 'slipway', 'social', 'spices', 'spring', 'stable', 'static', 'stationery', 'stop', 'store', 'substation', 'supplements', 'supply', 'synagogue', 'table', 'tattoo', 'tea', 'telephone', 'tent', 'terrace', 'ticket', 'tiles', 'tobacco', 'toilets', 'tower', 'townhall', 'toys', 'trade', 'trailer', 'transformer', 'tyres', 'vacuum', 'variety', 'vehicle', 'vending', 'video', 'wash', 'waste', 'watches', 'water', 'weapons', 'wholesale', 'window', 'wine', 'wool', 'yard', 'yes', 'artwork', 'baby', 'bag', 'bakery', 'bar', 'bath', 'books', 'bowling', 'bungalow', 'castle', 'conservatory', 'court', 'de', 'dog', 'fashion', 'garage', 'garages', 'motorcycle', 'nursing', 'rink', 'trophy'}

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

print(len(b_gdf))
b_gdf.head()

# %% [markdown]
# From: https://chat.openai.com/c/53d36cdf-6a9d-41e1-aa9f-336adb8141a7
# 
# Please sort the words in the following list into these 3 categories "work" "leisure" "errand".
# For example the word "bank" from the list could be in the category "work" and "errand"
# Please provide the category assignment as 3 python dictionaries with the name "work" "leisure" and "errand". Please provide only the 3 sets as answer no additional code is needed. Please make sure to use all the words form the word_list.

# %%
categories_set = set()
for categorie in b_gdf["categories"]:
    categories_set = categories_set.union(set(categorie))

gpt_all = work_set.union(leisure_set).union(errand_set)
print(gpt_all.difference(categories_set))
print(categories_set.difference(gpt_all))

print(len(gpt_all), len(categories_set))

# %%
b_gdf[["work","leisure", "errand"]].sum()

# %%
print(b_gdf[["work","leisure", "errand"]].sum())
sns.barplot(b_gdf[["work","leisure", "errand"]].sum())

# %%
folium_map = plot_regions(area, colormap=["rgba(0,0,0,0.1)"], tiles_style="CartoDB positron")

tooltip = folium.GeoJsonTooltip(
    fields=["osm_id", "work","leisure", "errand"],
    labels=True,
)


# show a sample of all geoms, all would crush the kernel
folium.GeoJson(b_gdf[["geom","osm_id", "work","leisure", "errand"]].sample(n=20000, random_state=1), tooltip=tooltip).add_to(folium_map)
folium_map

# %%
# TODO: Merge POI info with end of trajectory
pnt = gdf.iloc[0]["end"]
min_id = b_gdf["geom"].apply(lambda x: x.distance(pnt)).idxmin()

# %%
# takes >= 40min
def point_to_closest_osm_building(pnt):
    min_id = b_gdf["geom"].apply(lambda x: x.distance(pnt)).idxmin()
    return b_gdf[b_gdf.index==min_id]["osm_id"].iloc[0]

gdf["osm_end"] = gdf["end"].apply(point_to_closest_osm_building)
gdf.head()

# %%
gdf.columns

# %%
# get all hexagons of munich
# assigne pois to them
# plot result


