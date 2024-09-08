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
import torch
import sys

sys.path.append('..')
sys.path.append('../../')

# %%

from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
import numpy as np
from torch_geometric_temporal.signal import temporal_signal_split
from iteration_utilities import flatten

from master_thesis.utils.MyTransform import MyTransform
from master_thesis.utils.TemporalGNN import TemporalGNN
from master_thesis.utils.utils import visualize_single_prediction_label_pairs
from master_thesis.utils.utils import visualize_all_prediction_label_pairs

from master_thesis.utils.utils import intersection_over_union
from master_thesis.utils.utils import get_distances_df
from master_thesis.utils.utils import radius_of_gyration, get_jump_sizes, k_radius_of_gyration
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# %load_ext autoreload
# %autoreload 2

# %%
figsize_small=(10.5/2.54, 7.25/2.54)
figsize_large=(15.5/2.54, 10/2.54)
figsize_medium=(15.5/2.54, 7/2.54)

fontsize_small = 6
fontsize_medium = 8
fontsize_large = 12
dpi = 1000

# %%
device = "cpu"

# %%
resolution = 8
tsin = 3
tsout = 3

dataset_name = f"dataset_res{resolution}"
loader = TrajectoryDatasetLoader(f"../data/dataset_res{resolution}_v2.json", tsin, tsout)
dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)
train_snapshots_count = int(0.8 * dataset.snapshot_count)
sample_to_user_id = loader.sample_to_user_id

ids_to_hex_names = loader.ids_to_hex_names
groundtruth = torch.stack([snapshot.y for snapshot in test_set])

groundtruth.shape

# %%
regions = pd.read_pickle(f"../data/regions_enhanced_res{8}_v2.pkl")
distances_df = get_distances_df(regions)
regions.head()

# %%
baseline = torch.load(f"../results/baselie_res8_tsin3_tsout3_ns1119.pt")
baseline.shape

# %%
best_model_name = "model_rgclA3TGCN_oc64_ld32_lr0.1_do0.7_in3_out3_dsdataset_res8_v2_ns5595_ep50"

node_features = 7
tin = 3
tout = 3

config = {"rgcl": "A3TGCN", "oc": 64, "ld": 32, "do": 0.7 ,"in": 3, "out": 3, "nf": 7}
model = TemporalGNN(config)
model = model.to(device)

checkpoint = torch.load("../results/models_top5_v3/"+best_model_name +".pth") 
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# %%
model_predictions = []
model_out = []
snapshot = next(iter(test_set))
edge_index = snapshot.edge_index
edge_attr = snapshot.edge_attr
test_transform = MyTransform(test_set, 0,1)

for snapshot in test_set:
    with torch.no_grad():
        x = snapshot.x
        x = test_transform.min_max_scale(x).float()
        y_hat = model(x, edge_index, edge_attr)
        model_out.append(y_hat)

#threshold = 0.178261
threshold = 0.178426
model_out = torch.stack(model_out)
model_predictions = torch.sigmoid(model_out)
learned01 = (model_predictions > threshold).to(torch.int64)

learned01.shape

# %%
#torch.save(model_predictions01,"../results/learned01_res8_tsin3_tsout3_ns1119.pt" )

# %%
#torch.save(model_predictions,"../results/learnedsig_res8_tsin3_tsout3_ns1119.pt" )

# %%
#torch.save(ground_truth,"../results/groundtruth_res8_tsin3_tsout3_ns1119.pt" )
#torch.save(model_predictions01,"../results/learned_res8_tsin3_tsout3_ns1119.pt" )

# %% [markdown]
# ### Visual inspection

# %%
import folium
import branca.colormap as cm

def visualize_single_triphistory(trip_history, ids_to_hex_names, regions):

    visualize_df = regions.copy()
    for i in range(trip_history.shape[0]):
        visualize_df[f"trip{i}"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(trip_history[i,:].int()) }
       
    cmap = cm.StepColormap(["white", "blue"], vmin=0, vmax=1) 
    map = folium.Map(location=(48.12, 11.58), zoom_start=10)

    for i in range(trip_history.shape[0]):
        visualize_df.explore(name=f"trip{i}", m=map, cmap=cmap, column=f"trip{i}", style_kwds = {"fillOpacity" : 0.2})
        
    folium.LayerControl().add_to(map)
    return map

# %%
example_id = 14
# 3 -> okay
# 8 -> okay
# 22 -> bad
# 14 -> good
# 48 -> bad
# 18 -> bad
# 7 -> bad
base_line_example = baseline[example_id]
ground_truth_example = groundtruth[example_id]
model_predictions_example = learned01[example_id]

base_line_example_t = base_line_example.T
ground_truth_example_t = ground_truth_example.T
model_predictions_example_t = model_predictions_example.T

# %%
#visualize_single_prediction_label_pairs(model_predictions_example_t, ground_truth_example_t, ids_to_hex_names, regions)
#visualize_single_prediction_label_pairs(base_line_example_t, ground_truth_example_t, ids_to_hex_names, regions)
#visualize_single_prediction_label_pairs(model_predictions_example_t, ground_truth_example_t, ids_to_hex_names, regions)

# %%
# get all trips of one person

# %%

# %%
base_line_example_t = base_line_example.T
ground_truth_example_t = ground_truth_example.T
model_predictions_example_t = model_predictions_example.T

visualize_single_triphistory(model_predictions_example_t, ids_to_hex_names, regions)

# %%
base_line_sm = torch.sum(baseline, dim=(0,2))
ground_truth_sm = torch.sum(groundtruth, dim=(0,2))
model_predictions_sm = torch.sum(model_predictions, dim=(0,2))

# %%
def visualize_all_triphistories(trips, ids_to_hex_names, regions):
    visualize_df = regions.copy()
    visualize_df["trips"] = {ids_to_hex_names[i]: int(v) for i,v in enumerate(trips.int()) }
    max_visits = visualize_df["trips"].quantile(0.9) # max()
    cmap_labels = cm.LinearColormap(["white", "blue"], vmin=0, vmax=max_visits)
    map = folium.Map(location=(48.12, 11.58), tiles=None, zoom_start=10)
    folium.TileLayer("openstreetmap").add_to(map)
    visualize_df.explore(name=f"trips",     m=map, cmap=cmap_labels, column=f"trips",   style_kwds = {"fillOpacity" : 0.8},)
    folium.LayerControl().add_to(map)
    return map

# %%
visualize_all_triphistories(model_predictions_sm, ids_to_hex_names, regions)

# %%

#visualize_all_prediction_label_pairs(ground_truth_sm,model_predictions_sm, ids_to_hex_names, regions)
#visualize_all_prediction_label_pairs(ground_truth_sm, base_line_sm, ids_to_hex_names, regions)

# %%
# differences

# %% [markdown]
# ### Statistical values

# %%
baseline = torch.load("../results/baselie_res8_tsin3_tsout3_ns1119.pt")
learned01 = torch.load("../results/learned01_res8_tsin3_tsout3_ns1119.pt")
groundtruth = torch.load("../results/groundtruth_res8_tsin3_tsout3_ns1119.pt")

# %%
baseline.shape, groundtruth.shape, learned01.shape

model_01_np = learned01.int().numpy()
ground_truth_np = groundtruth.int().numpy()
base_line_np = baseline.int().numpy()

model_01_flat = model_01_np.flatten()
ground_truth_flat = ground_truth_np.flatten()
base_line_flat = base_line_np.flatten()

# %%
labels = ["not visited", "visited"]
cm_ml = confusion_matrix(ground_truth_flat, model_01_flat, normalize="all")
cm_bl = confusion_matrix(ground_truth_flat, base_line_flat, normalize="all")

cm_ml,cm_bl

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_small)

#fmt 3 nachkomma stellen
ax0 = sns.heatmap(cm_ml, annot=True, xticklabels=labels, square=True,yticklabels=labels, cmap="coolwarm", ax=axes[0],cbar=False)
ax1 = sns.heatmap(cm_bl, annot=True,  xticklabels=labels, square=True,yticklabels=labels, cmap="coolwarm", ax=axes[1],cbar=False)


ax0.set_xlabel("Learned Prediction Normalized", fontsize = fontsize_medium)
ax0.set_ylabel("Ground Truth Normalized", fontsize = fontsize_medium)

ax1.set_xlabel("Baseline Prediction Normalized", fontsize = fontsize_medium)
ax1.set_ylabel("", fontsize = fontsize_medium)

ax0.set_title("", fontsize = fontsize_medium)
ax1.set_title("", fontsize = fontsize_medium)

ax0.tick_params(axis='x', labelsize=fontsize_medium)
ax0.tick_params(axis='y', labelsize=fontsize_medium)
ax1.tick_params(axis='x', labelsize=fontsize_medium)
ax1.tick_params(axis='y', labelsize=fontsize_medium)
plt.tight_layout()

fig.savefig('../plots/evaluation/confusion_matrix_report.png', dpi=dpi)

plt.show()

# %%
class_report_ml_dict = classification_report(ground_truth_flat,model_01_flat, target_names=labels,output_dict=True)
class_report1_df = pd.DataFrame(class_report_ml_dict)
class_report1_df

# %%
class_report_bl_dict = classification_report(ground_truth_flat,base_line_flat, target_names=labels,output_dict=True)
class_report2_df = pd.DataFrame(class_report_bl_dict)
class_report2_df

# %%
#https://medium.com/@Doug-Creates/plotting-scikit-learn-classification-report-for-analysis-0229447fe232
fig, ax = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium)

report_ml = class_report_ml_dict
report_bl = class_report_bl_dict
labels = list(report_ml.keys())[:-3]  # Exclude 'accuracy', 'macro avg', 'weighted avg'
labels = labels[::-1]
metrics = ['precision', 'recall', 'f1-score']

data_ml = np.array([[report_ml[label][metric] for metric in metrics] for label in labels])
data_bl = np.array([[report_bl[label][metric] for metric in metrics] for label in labels])

# Blues
cax0 = ax[0].matshow(data_ml, cmap='coolwarm')
cax1 = ax[1].matshow(data_bl, cmap='coolwarm')

for (i, j), val in np.ndenumerate(data_ml):
        ax[0].text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize= fontsize_medium)

for (i, j), val in np.ndenumerate(data_bl):
        ax[1].text(j, i, f'{val:.2f}', ha='center', va='center', color='white', fontsize= fontsize_medium)

ax[0].set_xticks(range(len(metrics)), metrics, fontsize= fontsize_medium)
ax[0].set_yticks(range(len(labels)), labels, fontsize= fontsize_medium)

ax[1].set_xticks(range(len(metrics)), metrics, fontsize= fontsize_medium)
ax[1].set_yticks(range(len(labels)), ["", ""], fontsize= fontsize_medium)


ax[0].set_xlabel('Classification Report Learned', fontsize= fontsize_medium)
ax[0].set_ylabel('Classes', fontsize= fontsize_medium)

ax[1].set_xlabel('Classification Report Baseline', fontsize= fontsize_medium)
ax[1].set_ylabel('')

ax[0].set_title('', fontsize= fontsize_medium)
ax[1].set_title('', fontsize= fontsize_medium)

plt.tight_layout()
fig.savefig('../plots/evaluation/classification_report.png', dpi=dpi)

# %%
# TODO: make small plot


#weight_rebal = get_weight_rebal(train_set)
criterion = torch.nn.BCELoss()
criterion(learned01.float(), groundtruth.float()).item(), criterion(baseline.float(), groundtruth.float()).item(),  criterion(groundtruth.float(), groundtruth.float()).item()

# %% [markdown]
# ### Mobility values

# %%
groundtruth.sum(), baseline.sum(),learned01.sum()

# %%
user_gt_dict = dict()
user_ml_dict = dict()
user_bl_dict = dict()

for i,(gt_ss,bl_ss, ml_ss ) in enumerate(zip(groundtruth, baseline,learned01)):
    uid = sample_to_user_id[i+train_snapshots_count]
    
    if uid not in user_gt_dict.keys():
        user_gt_dict[uid] = []
        user_ml_dict[uid] = []
        user_bl_dict[uid] = []

    user_gt_dict[uid].append(gt_ss)
    user_bl_dict[uid].append(bl_ss)
    user_ml_dict[uid].append(ml_ss)

# %%
gt_rogs_dict = {uid: radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_gt_dict.items()}
bl_rogs_dict = {uid: radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_bl_dict.items()}
ml_rogs_dict = {uid: radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_ml_dict.items()}

# %%
gt_krogs_dict= {uid: k_radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_gt_dict.items()}
bl_krogs_dict= {uid: k_radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_bl_dict.items()}
ml_krogs_dict= {uid: k_radius_of_gyration(trip_histories,ids_to_hex_names, distances_df)for uid,trip_histories in user_ml_dict.items()}

# %%
gt_jmpss_ps = [get_jump_sizes(trip_histories,ids_to_hex_names, distances_df) for trip_histories in groundtruth]
bl_jmpss_ps = [get_jump_sizes(trip_histories,ids_to_hex_names, distances_df) for trip_histories in baseline]
ml_jmpss_ps = [get_jump_sizes(trip_histories,ids_to_hex_names, distances_df) for trip_histories in learned01]

gt_jmpss = list(flatten(gt_jmpss_ps))
bl_jmpss = list(flatten(bl_jmpss_ps))
ml_jmpss = list(flatten(ml_jmpss_ps))

# %%
gt_njmpss = []
bl_njmpss = []
ml_njmpss = []


for sample_id, uid in sample_to_user_id.items():
    if sample_id < train_snapshots_count:
        continue

    for js in gt_jmpss_ps[sample_id-train_snapshots_count]:
        gt_njmpss.append(js/gt_rogs_dict[uid]if gt_rogs_dict[uid] > 0 else 0)
    for js in bl_jmpss_ps[sample_id-train_snapshots_count]:
        bl_njmpss.append(js/bl_rogs_dict[uid] if bl_rogs_dict[uid] > 0 else 0)
    for js in ml_jmpss_ps[sample_id-train_snapshots_count]:
        ml_njmpss.append(js/ml_rogs_dict[uid]if ml_rogs_dict[uid] > 0 else 0)

# %%
gt_rog_df = pd.DataFrame({'rog': list(gt_rogs_dict.values()), "krog":list(gt_krogs_dict.values())})
bl_rog_df = pd.DataFrame({'rog': list(bl_rogs_dict.values()), "krog":list(bl_krogs_dict.values())})
ml_rog_df = pd.DataFrame({'rog': list(ml_rogs_dict.values()), "krog":list(ml_krogs_dict.values())})

rog_df = pd.concat([gt_rog_df.assign(Source='Groundtruth'), bl_rog_df.assign(Source='Baseline'), ml_rog_df.assign(Source='Learned')])

# %%
gt_jmps_df = pd.DataFrame({'jmps': gt_jmpss, "njmps":gt_njmpss})
bl_jmps_df = pd.DataFrame({'jmps': bl_jmpss, "njmps":bl_njmpss})
ml_jmps_df = pd.DataFrame({'jmps': ml_jmpss, "njmps":ml_njmpss})



# %%
jmps_df = pd.concat([gt_jmps_df.assign(Source='Groundtruth'), bl_jmps_df.assign(Source='Baseline'), ml_jmps_df.assign(Source='Learned')])


# %%
print(rog_df.groupby("Source").describe().T.to_latex())

# %%
print(jmps_df.groupby("Source").describe().T.to_latex())

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

ax0 = sns.histplot(data=rog_df, x="rog", hue="Source", multiple="dodge",ax=axes[0])
axes[0].title.set_size(fontsize_medium)


ax1 = sns.histplot(data=rog_df, x="krog", hue="Source",multiple="dodge", ax=axes[1])
axes[1].title.set_size(fontsize_medium)

axes[0].set_xlabel("Radius of Gyration [m]", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("K-Radius of Gyration [m]", fontsize=fontsize_medium)
axes[1].set_ylabel("")


axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)


axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)

axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

ax0.legend(["Learned", "Baseline","Groundtruth"], fontsize=fontsize_medium)
ax1.legend(["Learned", "Baseline","Groundtruth"], fontsize=fontsize_medium)

axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[1].set_xlim(0,axes[1].get_xlim()[1])

plt.tight_layout()

#fig.savefig('../plots/evaluation/synthetic_rog_compare.png', dpi=dpi)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium )

ax0 = sns.histplot(data=jmps_df, x="jmps", hue="Source", multiple="dodge")

ax0.set_xlabel("Jump Size [m]", fontsize=fontsize_medium)
ax0.set_ylabel("Count", fontsize=fontsize_medium)


ax0.set_title("", fontsize=fontsize_medium)

ax0.tick_params(axis='x', labelsize=fontsize_medium)
ax0.tick_params(axis='y', labelsize=fontsize_medium)

ax0.legend(["Learned", "Baseline","Groundtruth"], fontsize=fontsize_medium)

ax0.set_xlim(0,ax0.get_xlim()[1])

plt.tight_layout()

fig.savefig('../plots/evaluation/synthetic_jumpsize_compare2.png', dpi=dpi)

# %%
fig, axes = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium )

ax0 = sns.histplot(data=jmps_df, x="jmps", hue="Source", multiple="dodge",ax=axes[0])
ax1 = sns.histplot(data=jmps_df, x="njmps", hue="Source",multiple="dodge", ax=axes[1])

axes[0].set_xlabel("Jump Size [m]", fontsize=fontsize_medium)
axes[0].set_ylabel("Count", fontsize=fontsize_medium)
axes[1].set_xlabel("Normalized Jump Size", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)

axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

ax0.legend(["Learned", "Baseline","Groundtruth"], fontsize=fontsize_medium)
ax1.legend(["Learned", "Baseline","Groundtruth"], fontsize=fontsize_medium)

axes[0].set_xlim(0,axes[0].get_xlim()[1])
axes[1].set_xlim(0,axes[1].get_xlim()[1])

plt.tight_layout()

#fig.savefig('../plots/evaluation/synthetic_jumpsize_compare.png', dpi=dpi)

# %%
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize_medium )

ax = sns.scatterplot(data=rog_df, x="rog",y="krog", hue="Source", ax=ax, s= 15)
ax.set_xlabel("Radius of Gyration [m]", fontsize=fontsize_medium)
ax.set_ylabel("K-Radius of Gyration [m]", fontsize=fontsize_medium)

ax.set_title("", fontsize=fontsize_medium)

ax.tick_params(axis='x', labelsize=fontsize_medium)
ax.tick_params(axis='y', labelsize=fontsize_medium)

ax.legend(fontsize=fontsize_medium)

ax.set_xlim(0,ax.get_xlim()[1])
ax.set_ylim(0,ax.get_ylim()[1])

plt.tight_layout()
#fig.savefig('../plots/evaluation/synthetic_rogvskrog_compare.png', dpi=dpi)



