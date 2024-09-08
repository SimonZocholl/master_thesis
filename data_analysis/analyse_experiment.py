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
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from master_thesis.utils.TemporalGNN import TemporalGNN
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.utils import add_median_labels
import torch

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import numpy as np

# %load_ext autoreload
# %autoreload 2

# %%
figsize_small=(10.5/2.54, 7.25/2.54)
figsize_large=(15.5/2.54, 10/2.54)
figsize_medium=(15.5/2.54, 7/2.54)

fontsize_small = 6
fontsize_medium = 8
fontsize_large = 12
plt.rcParams['savefig.dpi'] = 1000
dpi = 1000

# %%
#experiment_improved.json

results_df  = pd.read_json("../results/results_hp4_v2.json").T

results_df["ld"] = results_df["ld"].astype(int)
results_df["oc"] = results_df["oc"].astype(int)

results_df["lr"] = results_df["lr"].astype(float)
results_df["do"] = results_df["do"].astype(float)

results_df["in"] = results_df["in"].astype(float)
results_df["out"] = results_df["out"].astype(float)

results_df["eps"] = results_df["eps"].astype(int)
results_df["nf"] = results_df["nf"].astype(int)
results_df["ns"] = results_df["ns"].astype(int)

results_df["bcew"] = results_df["bcew"].astype(float)
results_df["bce"] = results_df["bce"].astype(float)
results_df["best_threshold"] = results_df["best_threshold"].astype(float)

results_df["fscore"] = results_df["fscore"].astype(float)
results_df["auc"] = results_df["auc"].astype(float)
results_df["res"] = 8

results_df.head()

# %%
results_df.reset_index().head()

# %%
table_features = ["rgcl", "ld", "oc", "do", "in", "out", "ns" , "eps", "bce" , "best_threshold"]
#print(results_df.reset_index()[table_features].to_latex())

# %%
content = []

for key, val in results_df.iterrows():
    for i,tl in enumerate(val["train_losses"]):
        content.append([key, "Train",tl,i])
        
    for i,vl in enumerate(val["val_losses"]):
        content.append([key, "Test",vl, i])

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

ax0 = sns.barplot(data=results_df.sort_values("bce"), y="bce", x =results_df.index.values, ax=axes[0])
ax0.set_xticklabels("")


results_hp_train_df = pd.DataFrame(content, columns=["model", "source", "bce", "epoch"])
ax1 = sns.lineplot(data=results_hp_train_df, y = "bce", x = "epoch",hue = "source",  ax=axes[1])

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)


axes[0].set_xlabel("Final Test Performance", fontsize=fontsize_medium)
axes[0].set_ylabel("BCE", fontsize=fontsize_medium)
axes[1].set_xlabel("Average Performance over Epochs", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

axes[1].set_xlim(0,axes[1].get_xlim()[1])

plt.legend(fontsize=fontsize_medium)
plt.tight_layout()

#fig.savefig('../plots/evaluation/model_hp_performances.png', dpi=dpi)

# %%
results_bad_df = results_df[results_df["bce"] >= 0.1]
#print(results_bad_df[["oc", "do", "lr"]].value_counts().to_latex())

# %%
fig, axes = plt.subplots(nrows=2, ncols=2,figsize=(figsize_large[0], figsize_large[1]*1.5))

ax00 = sns.boxplot(data=results_df, y="bce", x ="rgcl", ax=axes[0,0])
add_median_labels(ax00,fmt=".3f",fontsize=fontsize_medium)

ax01 = sns.boxplot(data=results_df, y="bce", x ="oc", ax=axes[0,1])
add_median_labels(ax01,fmt=".3f",fontsize=fontsize_medium)

ax10 = sns.boxplot(data=results_df, y="bce", x ="do", ax=axes[1,0])
add_median_labels(ax10,fmt=".3f",fontsize=fontsize_medium)

ax11 = sns.boxplot(data=results_df, y="bce", x ="lr", ax=axes[1,1])
add_median_labels(ax11,fmt=".3f",fontsize=fontsize_medium)

for i,j in [[0,0],[1,0], [0,1],[1,1]]:
    axes[i,j].tick_params(axis='x', labelsize=fontsize_medium)
    axes[i,j].tick_params(axis='y', labelsize=fontsize_medium)
    axes[i,j].xaxis.label.set_size(fontsize_medium)
    axes[i,j].yaxis.label.set_size(fontsize_medium)

axes[0,1].set_ylabel("")
axes[1,1].set_ylabel("")

axes[0,0].set_ylabel("BCE",fontsize=fontsize_medium)
axes[1,0].set_ylabel("BCE",fontsize=fontsize_medium)

axes[1,0].set_xticklabels([0.3, 0.5, 0.7])

axes[0,0].set_xlabel(f"RGCL", fontsize=fontsize_medium)
axes[0,1].set_xlabel(f"Output Channel", fontsize=fontsize_medium)
axes[1,0].set_xlabel(f"Dropout", fontsize=fontsize_medium)
axes[1,1].set_xlabel(f"Learning Rate", fontsize=fontsize_medium)

plt.tight_layout()

#fig.savefig('../plots/evaluation/hparam_performances.png', dpi=dpi)

# %% [markdown]
# # TOP5 Evaluation !!!
#

# %%
results_top5_df  = pd.read_json("../results/result_top5_v3.json").T

results_top5_df["ld"] = results_top5_df["ld"].astype(int)
results_top5_df["oc"] = results_top5_df["oc"].astype(int)

results_top5_df["lr"] = results_top5_df["lr"].astype(float)
results_top5_df["do"] = results_top5_df["do"].astype(float)

results_top5_df["in"] = results_top5_df["in"].astype(float)
results_top5_df["out"] = results_top5_df["out"].astype(float)

results_top5_df["eps"] = results_top5_df["eps"].astype(int)
results_top5_df["nf"] = results_top5_df["nf"].astype(int)
results_top5_df["ns"] = results_top5_df["ns"].astype(int)

results_top5_df["bcew"] = results_top5_df["bcew"].astype(float)
results_top5_df["bce"] = results_top5_df["bce"].astype(float)
results_top5_df["best_threshold"] = results_top5_df["best_threshold"].astype(float)

results_top5_df["fscore"] = results_top5_df["fscore"].astype(float)
results_top5_df["auc"] = results_top5_df["auc"].astype(float)

results_top5_df.head()

# %%
print(results_top5_df.nsmallest(5,"bce").reset_index()[table_features].to_latex())

# %%
best_model_top5 = results_top5_df[results_top5_df["bce"] == results_top5_df["bce"].min()]
best_model_top5

# %%
content = []
for key, val in results_top5_df.iterrows():
    for i,tl in enumerate(val["train_losses"]):
        content.append([key, "Train",tl,i])
        
    for i,vl in enumerate(val["val_losses"]):
        content.append([key, "Test",vl, i])

# TODO rename index to epochs
results_top5_train_df = pd.DataFrame(content, columns=["model", "source", "bce", "epoch"])


# %%
#df1 = pd.DataFrame({"losses": best_model_top5["train_losses"][0]})
#df2 = pd.DataFrame({"losses": best_model_top5["val_losses"][0]})
#best_model_top5_train_df = pd.concat([df1.assign(source='train'), df2.assign(source='val')])
#sns.lineplot(data=best_model_top5_train_df, y="losses", x =best_model_top5_train_df.index.values, hue="source")

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )

ax0 = sns.barplot(data=results_top5_df.sort_values("bce"), y="bce", x =results_top5_df.index.values, ax=axes[0])
#ax0.bar_label(ax0.containers[0])


ax0.set_xticklabels("")


results_hp_train_df = pd.DataFrame(results_top5_train_df, columns=["model", "source", "bce", "epoch"])
ax1 = sns.lineplot(data=results_hp_train_df, y = "bce", x = "epoch",hue = "source",  ax=axes[1])
ax1.set_ylim([0, 3.2])

axes[0].set_title("", fontsize=fontsize_medium)
axes[1].set_title("", fontsize=fontsize_medium)


axes[0].set_xlabel("Final Test Performance", fontsize=fontsize_medium)
axes[0].set_ylabel("BCE", fontsize=fontsize_medium)
axes[1].set_xlabel("Average Performance over Epochs", fontsize=fontsize_medium)
axes[1].set_ylabel("")
axes[0].set_xticklabels([0,1,2,3,4,5])

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

ax0.bar_label(ax0.containers[0], fontsize=fontsize_medium, fmt='%.3f')

axes[0].set_ylim(0,axes[0].get_ylim()[1]*1.1)
axes[1].set_ylim(0,axes[1].get_ylim()[1]*1.1)

plt.legend(fontsize=fontsize_medium)
plt.tight_layout()

#fig.savefig('../plots/evaluation/model_top5_performances.png', dpi = dpi)

# %%
df1 = pd.DataFrame({"BCE": best_model_top5["train_losses"][0]})
df2 = pd.DataFrame({"BCE": best_model_top5["val_losses"][0]})
best_model_top5_train = pd.concat([df1.assign(Source='Train'), df2.assign(Source='Test')])

sns.lineplot(data=best_model_top5_train, y="BCE", x =best_model_top5_train.index.values, hue="Source")

# %%
device = "cpu"
dataset_name = f"dataset_res{8}"
loader = TrajectoryDatasetLoader(f"../data/dataset_res{8}_v2.json", 3, 3)
dataset = loader.get_dataset()
snapshot = next(iter(dataset))
edge_index = snapshot.edge_index
edge_attr = snapshot.edge_attr

# %%
best_model_name = "model_rgclA3TGCN_oc64_ld32_lr0.1_do0.7_in3_out3_dsdataset_res8_v2_ns5595_ep50"

node_features = 7
tin = 3
tout = 3

config = {"rgcl": "A3TGCN", "oc": 64, "ld": 32, "do": 0.5 ,"in": 3, "out": 3, "nf": 7}
model = TemporalGNN(config).to(device)
model

# %%
predictionss_sig = torch.load("../results/learnedsig_res8_tsin3_tsout3_ns1119.pt")
ground_truth = torch.load("../results/groundtruth_res8_tsin3_tsout3_ns1119.pt")

# %%
labels_arr_flatt = (ground_truth).flatten().detach().numpy()
predictions_arr_flatt = (predictionss_sig).flatten().detach().numpy()
precision, recall, thresholds =precision_recall_curve(labels_arr_flatt, predictions_arr_flatt)

# %%
# convert to f score
epsilon = 1e-10*np.ones_like(precision)
fscore = (2 * precision * recall) / (precision + recall +epsilon)
# locate the index of the largest f score
ix = np.argmax(fscore)
best_threshold =  thresholds[ix]
best_fscore = fscore[ix]
auc_score = auc(recall, precision)

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium )
ax0 = sns.lineplot(data=best_model_top5_train, y="BCE", x =best_model_top5_train.index.values, hue="Source", ax=axes[0])

ax1 = sns.lineplot(x=precision, y=recall, ax=axes[1])
sns.scatterplot(x=[precision[ix]], y=[recall[ix]], color='red', ax=axes[1])

ax1.set_ylim([0,1])
ax1.set_xlim([0,1])


axes[0].set_xlabel("Best Model Training Performance", fontsize=fontsize_medium)
axes[0].set_ylabel("BCE", fontsize=fontsize_medium)
axes[1].set_xlabel("Precision", fontsize=fontsize_medium)
axes[1].set_ylabel("Recall", fontsize=fontsize_medium)

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

axes[0].set_xlim(0,axes[0].get_xlim()[1])

plt.legend(fontsize=fontsize_medium)
plt.tight_layout()

fig.savefig('../plots/evaluation/best_model_train_recal_precision.png', dpi = dpi)

# %%
fig.savefig('../plots/evaluation/best_model_train_recal_precision.png', dpi = dpi)

# %% [markdown]
# # In Outs

# %%
results_inout_df  = pd.read_json("../results/results_inouts_v3.json").T

results_inout_df["ld"] = results_inout_df["ld"].astype(int)
results_inout_df["oc"] = results_inout_df["oc"].astype(int)

results_inout_df["lr"] = results_inout_df["lr"].astype(float)
results_inout_df["do"] = results_inout_df["do"].astype(float)

results_inout_df["in"] = results_inout_df["in"].astype(int)
results_inout_df["out"] = results_inout_df["out"].astype(int)

results_inout_df["eps"] = results_inout_df["eps"].astype(int)
results_inout_df["nf"] = results_inout_df["nf"].astype(int)
results_inout_df["ns"] = results_inout_df["ns"].astype(int)

results_inout_df["bcew"] = results_inout_df["bcew"].astype(float)
results_inout_df["bce"] = results_inout_df["bce"].astype(float)
results_inout_df["best_threshold"] = results_inout_df["best_threshold"].astype(float)

results_inout_df["fscore"] = results_inout_df["fscore"].astype(float)
results_inout_df["auc"] = results_inout_df["auc"].astype(float)

results_inout_df.head()

# %%
results_res_df  = pd.read_json("../results/results_res_v3.json").T

results_res_df["ld"] = results_res_df["ld"].astype(int)
results_res_df["oc"] = results_res_df["oc"].astype(int)

results_res_df["lr"] = results_res_df["lr"].astype(float)
results_res_df["do"] = results_res_df["do"].astype(float)

results_res_df["in"] = results_res_df["in"].astype(int)
results_res_df["out"] = results_res_df["out"].astype(int)

results_res_df["eps"] = results_res_df["eps"].astype(int)
results_res_df["nf"] = results_res_df["nf"].astype(int)
results_res_df["ns"] = results_res_df["ns"].astype(int)

results_res_df["bcew"] = results_res_df["bcew"].astype(float)
results_res_df["bce"] = results_res_df["bce"].astype(float)
results_res_df["best_threshold"] = results_res_df["best_threshold"].astype(float)

results_res_df["fscore"] = results_res_df["fscore"].astype(float)
results_res_df["auc"] = results_res_df["auc"].astype(float)
results_res_df["res"] = [9,7,8]

results_res_df.head()

# %%
'''
fig, ax = plt.subplots(nrows=1, ncols=1,figsize=figsize_small)

ax = sns.boxplot(data=results_res_df, y="bce", x ="res", ax=ax)
add_median_labels(ax,fmt=".2f")

ax.set_title("", fontsize=fontsize_medium)

ax.set_xlabel("Resolution", fontsize=fontsize_medium)
ax.set_ylabel("BCE", fontsize=fontsize_medium)

ax.tick_params(axis='x', labelsize=fontsize_medium)
ax.tick_params(axis='y', labelsize=fontsize_medium)

fig.savefig('../plots/evaluation/resolution_performance.png')
'''

# %%
results_inout_df[["in", "out", "bce"]]

# %%
fig, axes = plt.subplots(nrows=1, ncols=2,figsize=figsize_medium)

ax0 = sns.lineplot(data=results_inout_df, y="bce", x ="in", ax=axes[0],errorbar=None)
sns.lineplot(data=results_inout_df, y="bce", x ="out", ax=axes[0],errorbar=None)

ax1 = sns.lineplot(data=results_res_df, y="bce", x ="res", ax=axes[1],errorbar=None)

axes[0].set_xlabel("Input Steps", fontsize=fontsize_medium)
axes[0].set_ylabel("BCE", fontsize=fontsize_medium)
#axes[1].set_xlabel("Output Steps", fontsize=fontsize_medium)
#axes[1].set_ylabel("")

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
#axes[1].tick_params(axis='x', labelsize=fontsize_medium)
#axes[1].tick_params(axis='y', labelsize=fontsize_medium)

#fig.savefig('../plots/evaluation/input_output_performance.png')
axes[0].set_title("", fontsize=fontsize_medium)
#axes[1].set_title("", fontsize=fontsize_medium)

axes[0].legend(["Input", "Output"],fontsize=fontsize_medium)
axes[0].set_xlabel("Input Output Steps", fontsize=fontsize_medium)
axes[0].set_ylabel("BCE", fontsize=fontsize_medium)
axes[1].set_xlabel("Resolution", fontsize=fontsize_medium)
axes[1].set_ylabel("")

axes[0].set_xticks([2,3,4])
axes[1].set_xticks([7,8,9])

axes[0].tick_params(axis='x', labelsize=fontsize_medium)
axes[0].tick_params(axis='y', labelsize=fontsize_medium)
axes[1].tick_params(axis='x', labelsize=fontsize_medium)
axes[1].tick_params(axis='y', labelsize=fontsize_medium)

plt.tight_layout()

fig.savefig('../plots/evaluation/input_output_res_performance.png', dpi=dpi)


