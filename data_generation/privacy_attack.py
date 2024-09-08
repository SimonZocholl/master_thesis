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
sys.path.append('..')
sys.path.append('../../')

# %%
import torch
import h3
import numpy as np
from tqdm import tqdm
from shapely import MultiPoint
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from torch_geometric_temporal.signal import temporal_signal_split
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader

# %load_ext autoreload
# %autoreload 2

# %%
def get_centroid_of_tripps(tripps,ids_to_hex_names):
    points = []
    for hex_id, hex_over_time in enumerate(tripps):
        for t in hex_over_time:
            if t:
                hex = ids_to_hex_names[hex_id]
                point = h3.cell_to_latlng(hex)
                points.append(point)

    centroid = MultiPoint(points).centroid
    return centroid

def get_centroids(public_dataset,visited_id,ids_to_hex_names):
    centroids = []
    
    for tripps in public_dataset[:,:,visited_id,:]:
        centroid = get_centroid_of_tripps(tripps,ids_to_hex_names)
        centroids.append(centroid)
    return centroids



def privacy_attack(private_dataset, published_dataset, centroids, sample_to_user_id, ids_to_hex_names, offset=4476):
    work_id = 3  #correct for user_id missing and messup
    leisure_id = 5 #correct for user_id missing
    errand_id = 4  #correct for user_id missing 
    visited_id = 6

    personal_data_ids = [work_id, leisure_id, errand_id]
    personalfit_candidate_dataset_counts = []
    true_position_in_candidates = []

    # Tripp to person
    for index in  tqdm(range(private_dataset.shape[0])):

        # Setup
        leaked_data = private_dataset[index]
        leaked_tripps = leaked_data[:,visited_id,:]
        leaked_personal = leaked_data[0,personal_data_ids,0]
        leaked_user_id = sample_to_user_id[index+offset]
        leaked_centroid = get_centroid_of_tripps(leaked_tripps,ids_to_hex_names)

        # candidates after personal fit
        personalfit_candidate_dataset_indices = []
        for i, personal in enumerate(published_dataset[:,0,personal_data_ids,0]):
            if torch.equal(personal, leaked_personal):
                personalfit_candidate_dataset_indices.append(i)
        # store result
        personalfit_candidate_dataset_counts.append(len(personalfit_candidate_dataset_indices))

        # closest by centroid distance
        centroid_distances = []
        for pf_d_i in personalfit_candidate_dataset_indices:
            centroid = centroids[pf_d_i]
            centroid_distance = leaked_centroid.distance(centroid)
            centroid_distances.append(centroid_distance)

        
        cent_dist_sorted_dataset_indices = np.argsort(centroid_distances)

        #print("sample_ids: ", some_ids)
        user_ids = [sample_to_user_id[personalfit_candidate_dataset_indices[sid]+offset] for sid in cent_dist_sorted_dataset_indices]
        index_user_id = user_ids.index(leaked_user_id)
        # store result
        true_position_in_candidates.append(index_user_id)

    return personalfit_candidate_dataset_counts, true_position_in_candidates

# %%
resolution = 8
tsin = 3
tsout = 3

dataset_name = f"dataset_res{resolution}"
loader = TrajectoryDatasetLoader(f"../data/dataset_res{resolution}_v2.json", tsin, tsout)
dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)
ids_to_hex_names = loader.ids_to_hex_names
offset = int(0.8 * dataset.snapshot_count)

sample_to_user_id = loader.sample_to_user_id

# %%
baseline = torch.load("../results/baselie_res8_tsin3_tsout3_ns1119.pt")
learned01 = torch.load("../results/learned01_res8_tsin3_tsout3_ns1119.pt")
groundtruth = torch.load("../results/groundtruth_res8_tsin3_tsout3_ns1119.pt")

# %%
# ["hexagon_work", "hexagon_leisure", "hexagon_errand", "user_work", "user_errand", "user_leisure", "visited"]

work_id = 3  #correct for user_id missing and messup
leisure_id = 5 #correct for user_id missing
errand_id = 4  #correct for user_id missing
visited_id = 6 #correct for user_id missing

# %%
dataset_private = torch.stack([snapshot.x for snapshot in test_set])

dataset_public_baseline = dataset_private
dataset_public_baseline[:,:,visited_id,:] = baseline

dataset_public_learned_ = dataset_private
dataset_public_learned_[:,:,visited_id,:] = learned01

dataset_groundtruth = dataset_private
dataset_groundtruth[:,:,visited_id,:] = groundtruth

dataset_groundtruth.shape

# %%
baseline_centroids = get_centroids(dataset_public_baseline,visited_id,ids_to_hex_names)
learned_centroids = get_centroids(dataset_public_learned_,visited_id,ids_to_hex_names)
groundtruth_centroids =get_centroids(dataset_groundtruth,visited_id,ids_to_hex_names)

# %%
def personal_attack_eval(leaked_personal, dataset_groundtruth, personal_data_ids,sample_to_user_id, offset):
    same_personal_indexs = []
    for i, personal in enumerate(dataset_groundtruth[:,0,personal_data_ids,0]):
            if torch.equal(personal, leaked_personal):
                 same_personal_indexs.append(sample_to_user_id[i+offset])
                    
    return len(set(same_personal_indexs))

def trip_attack_eval(index,leaked_tripps, centroids, ids_to_hex_names, sample_to_user_id, offset):
     # closest by centroid distance
    centroid_distances = []
    leaked_user_id = sample_to_user_id[index+offset]
    leaked_centroid = get_centroid_of_tripps(leaked_tripps,ids_to_hex_names)

    # closest by centroid distance
    centroid_distances = []
    for i in range(len(centroids)):
        centroid = centroids[i]
        centroid_distance = leaked_centroid.distance(centroid)
        centroid_distances.append(centroid_distance)

        
    centroid_distances_sorted_indices = np.argsort(centroid_distances)
    user_ids = [sample_to_user_id[sid+offset] for sid in centroid_distances_sorted_indices]
    return user_ids.index(leaked_user_id)

    

# %%
work_id = 3  #correct for user_id missing and messup
leisure_id = 5 #correct for user_id missing
errand_id = 4  #correct for user_id missing 
visited_id = 6

personal_data_ids = [work_id, leisure_id, errand_id]

personal_attack_res = []
true_position_in_candidates = []

trip_groundtruth_ress = []
trip_baseline_ress =  []
trip_learned_ress = []

# Tripp to person
for index in  tqdm(range(len(dataset_groundtruth))):
    leaked_personal = dataset_groundtruth[index,0,personal_data_ids,0]
    leaked_user_id = sample_to_user_id[index+offset]
    
    leaked_tripps = dataset_groundtruth[index,:,visited_id,:]
    #print(leaked_tripps.shape)
    # personal_attack_eval(leaked_personal, dataset_groundtruth, personal_data_ids,sample_to_user_id, offset)
    pers_res = personal_attack_eval(leaked_personal, dataset_groundtruth, personal_data_ids,sample_to_user_id, offset)
    personal_attack_res.append(pers_res)

    # trip_attack_eval(index,leaked_tripps, centroids, ids_to_hex_names)
    #trip_baseline_res = trip_attack_eval(index,leaked_tripps, baseline_centroids, ids_to_hex_names,  sample_to_user_id, offset)
    #trip_baseline_ress.append(trip_baseline_res)

    #trip_learned_res = trip_attack_eval(index,leaked_tripps, learned_centroids, ids_to_hex_names,  sample_to_user_id, offset)
    #trip_learned_ress.append(trip_learned_res)
    
    #trip_groundtruth_res = trip_attack_eval(index,leaked_tripps, groundtruth_centroids, ids_to_hex_names,  sample_to_user_id, offset)
    #trip_groundtruth_ress.append(trip_groundtruth_res)

    

# %%
len(personal_attack_res)

# %%
sum(trip_groundtruth_ress), sum(trip_learned_ress), sum(trip_baseline_ress)

# %%
cpfi_gt, tpic_gt = privacy_attack(dataset_private, dataset_groundtruth, groundtruth_centroids, sample_to_user_id, ids_to_hex_names)
cpfi_ml, tpic_ml = privacy_attack(dataset_private, dataset_public_learned_, learned_centroids, sample_to_user_id, ids_to_hex_names)
cpfi_bl, tpic_bl = privacy_attack(dataset_private, dataset_public_baseline, baseline_centroids, sample_to_user_id, ids_to_hex_names)

# %%
len(cpfi_gt), len(tpic_gt)

# %%
privacy_gt_df = pd.DataFrame({"personal_fits": cpfi_gt, "personal_pos": tpic_gt})
privacy_ml_df = pd.DataFrame({"personal_fits": cpfi_ml, "personal_pos": tpic_ml})
privacy_bl_df = pd.DataFrame({"personal_fits": cpfi_bl, "personal_pos": tpic_bl})

privacy_df = pd.concat([privacy_gt_df.assign(Source="Groundtruth"),
                        privacy_ml_df.assign(Source="Learned"),
                        privacy_bl_df.assign(Source="Baseline")])

# %%
figsize_small=(10.5/2.54, 7.25/2.54)
figsize_large=(15.5/2.54, 10/2.54)
figsize_medium=(15.5/2.54, 7/2.54)

fontsize_small = 6
fontsize_medium = 8
fontsize_large = 12

# %%
privacy_df2 = pd.DataFrame({"same_personal_features": personal_attack_res})

# %%
privacy_df2.value_counts()

# %%
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium)

ax0 = sns.histplot(privacy_df2, x="same_personal_features")

ax0.set_title("", fontsize=fontsize_medium)

ax0.tick_params(axis='x', labelsize=fontsize_medium)
ax0.tick_params(axis='y', labelsize=fontsize_medium)
ax0.legend(fontsize = fontsize_medium) 

ax0.set_xlabel("Persons with Same Personal Features", fontsize=fontsize_medium)
ax0.set_ylabel("Count", fontsize=fontsize_medium)

plt.tight_layout()

fig.savefig('../plots/evaluation/privacy_attack.png', dpi=1000)

# %%
print(privacy_df2.describe().T)

# %%
print(privacy_df2.describe().T.to_latex())


