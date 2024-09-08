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
import torch
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
from itertools import product

sys.path.append('..')
sys.path.append('../../')

# %load_ext autoreload
# %autoreload 2

# %%
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.MyTransform import MyTransform
from master_thesis.utils.TemporalGNN import TemporalGNN
from master_thesis.utils.utils import create_json_file, add_data_to_json
from master_thesis.utils.utils import get_pos_weights
from master_thesis.utils.utils import get_threshold_and_fscore
from master_thesis.utils.utils import train_loop, test_loop

# %%
print("start_script")

input_steps = 3
output_steps = 3
ns = 1000

# done
dataset_name = "dataset_res8_v2"
node_features = 7

loader = TrajectoryDatasetLoader(f"../data/{dataset_name}.json", 
                                 input_steps,
                                 output_steps,
                                 samples=ns)

dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)
train_transform = MyTransform(train_set, 0,1)
test_transform = MyTransform(test_set, 0,1)

# %%
device = torch.device('cpu')
if torch.cuda.is_available():
    device = "cuda:0"

# %%
# experiment
filename = "results_hp4_v2.json"

# https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html
pos_weight = get_pos_weights(train_set)
criterionw = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight).to(device)
criterion = torch.nn.BCEWithLogitsLoss().to(device)


dropp_out = [0.3, 0.5, 0.7]
out_channels = [2**i for i in range(T,6+3)]
lrs = [10**(-i) for i in range(1, 1+3)]
eps = 25
rgcls = ["GCLSTM","DCRNN", "A3TGCN"]

create_json_file(filename)
settings = list(product(rgcls, out_channels, lrs,dropp_out))

# %%
print("start training")
for rgcl, oc, lr,do  in tqdm(settings):
    
    ld = oc // 2
    model_name = f"model_rgcl{rgcl}_oc{oc}_ld{ld}_lr{lr}_do{do}_in{input_steps}_out{output_steps}_ds{dataset_name}_ns{ns}_ep{eps}"
    print(model_name)
    
    config = {
                "rgcl": rgcl,
                "ld": ld,
                "oc": oc,
                "lr" : lr,
                "do": do,
                "in": input_steps,
                "out": output_steps,
                "ds": dataset_name,
                "eps": eps,
                "nf": node_features,
                "ns": ns
            }
    
    model = TemporalGNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["eps"]
    # train_loop(model_name, model, criterion, optimizer, epochs, train_transform, test_transform, train_dataset,test_dataset)
    
    # TODO: now training with only BCE loss
    model,train_losses, val_losses = train_loop(model_name, 
                                                model,
                                                criterion, 
                                                optimizer,
                                                device,
                                                epochs, 
                                                train_transform, 
                                                test_transform, 
                                                train_set, 
                                                test_set)
    
    # test_loop(model, criterion, train_transform, test_dataset)
    predictions, labels = test_loop(model, train_transform, test_set, device)

    # calculate metrics
    labelss = torch.stack(labels).detach().cpu().numpy()
    predictionss = torch.stack(predictions).detach().cpu().numpy()

    predictionss_sig = torch.sigmoid(torch.tensor(predictionss))
    best_threshold, best_fscore, auc_score = get_threshold_and_fscore(labelss, predictionss_sig)
    
    bcew = criterionw(torch.tensor(predictionss).to(device),torch.tensor(labelss).to(device))
    bce = criterion(torch.tensor(predictionss).to(device),torch.tensor(labelss).to(device))

    result = { 
            "bcew": bcew.item(),
            "bce": bce.item(), 
            "best_threshold": best_threshold.item(),
            "fscore": best_fscore.item(), 
            "auc": auc_score.item(), 
            "train_losses": list(train_losses),
            "val_losses": list(val_losses)
            }
    
    entry = dict()
    entry[model_name] = config | result
    add_data_to_json(filename, entry)



