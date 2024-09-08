# %%
import sys
sys.path.append('..')
sys.path.append('../../')
#%load_ext autoreload
#%autoreload 2

# %%
import torch
from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
from itertools import product
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.MyTransform import MyTransform
from master_thesis.utils.TemporalGNN import TemporalGNN
from master_thesis.utils.utils import create_json_file, add_data_to_json
from master_thesis.utils.utils import get_pos_weights
from master_thesis.utils.utils import get_threshold_and_fscore
from master_thesis.utils.utils import train_loop, test_loop

# %%
print("start_script")

filename = "results_inout_res_v2.json"
create_json_file(filename)

device = torch.device('cpu')
if torch.cuda.is_available():
    device = "cuda:0"

resolutions = [7,8,9]
input_stepss = [2,3,4]
output_steps = [2,3,4]


do =  0.7
oc = 32
ld = 16
lr = 0.1
eps = 50
rgcl = "A3TGCN"
nf = 7


in_out_steps = list(product(input_stepss, output_steps, resolutions))

# %%
print("start training")
for ins, outs, res in tqdm(in_out_steps):
    print(f"in: {ins}, outs: {outs}, res: {res}")

    dataset_name = f"dataset_res{res}_v2"
    loader = TrajectoryDatasetLoader(f"../data/{dataset_name}.json", ins, outs)
    
    dataset = loader.get_dataset()
    
    train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)
    ns = train_set.snapshot_count + test_set.snapshot_count
    
    train_transform = MyTransform(train_set, 0,1)
    test_transform = MyTransform(test_set, 0,1)
    
    pos_weight = get_pos_weights(train_set)
    weight_rebal = get_pos_weights(train_set)
    criterion_weighted = torch.nn.BCEWithLogitsLoss(pos_weight=weight_rebal).to(device)
    criterion = torch.nn.BCEWithLogitsLoss().to(device)
    model_name = f"model_rgcl{rgcl}_oc{oc}_ld{ld}_lr{lr}_do{do}_in{ins}_out{outs}_ds{dataset_name}_ns{ns}_ep{eps}_nf{nf}"
    
    print(model_name)
    config = {
            "rgcl": rgcl,
            "ld": ld,
            "oc": oc,
            "lr" : lr,
            "do": do,
            "in": ins,
            "out": outs,
            "ds": dataset_name,
            "eps": eps,
            "nf": nf,
            "ns": ns
            }
    
    model = TemporalGNN(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
    epochs = config["eps"]
    # train_loop(model_name, model, criterion, optimizer, epochs, train_transform, test_transform, train_dataset,test_dataset)
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
    bcew = criterion_weighted(torch.tensor(predictionss).to(device),torch.tensor(labelss).to(device))
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




