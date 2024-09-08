# %%
import os
import torch
from itertools import product
import numpy as np

#sys.path.append('../')

from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm

from master_thesis.utils.TemporalGNN import TemporalGNN
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.MyTransform import MyTransform
from master_thesis.utils.utils import get_threshold_and_fscore
from master_thesis.utils.utils import create_json_file, add_data_to_json

%load_ext autoreload
%autoreload 2

# %%
def load_from_checkpoint(path, model, optimizer):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss

def get_weight_rebal(train_dataset):
    y = next(iter(train_dataset)).y
    num_elem = sum([snapshot.y.numel() for snapshot in train_dataset])
    num_ones = sum([snapshot.y.sum() for snapshot in train_dataset])
    factor = (num_elem-num_ones)/(num_ones)
    weight_rebal = torch.ones_like(y) / factor + (1.0 - 1.0/factor)*y
    weight_rebal.shape

# %%
def train_loop(name, model, criterion, optimizer,device, train_transform, test_transform,train_dataset, test_dataset, epochs):
     
    model.to(device)        
    model_paths = []

    model.train()
    snapshot = next(iter(train_dataset))
    edge_index = snapshot.edge_index.to(device)
    edge_attr = snapshot.edge_attr.to(device)

    train_losses = []
    val_losses = []

    for epoch in tqdm(range(epochs)): 
        train_loss = 0

        for _,snapshot in enumerate(train_dataset):
            x = snapshot.x
            y = snapshot.y.float().to(device)
            x = train_transform.min_max_scale(x).float().to(device)
            y_hat = model(x, edge_index, edge_attr)
            train_loss = train_loss + criterion(y_hat, y)
            
        train_loss = train_loss / train_dataset.snapshot_count
        train_losses.append(train_loss.item())

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # here we can add more scalars
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
            }

        # checkpoint saving
        model_dir = os.path.join(os.curdir, "checkpoints",name)
        model_path = os.path.join(model_dir,"epoch"+str(epoch)+".pth")
        model_paths.append(model_path)
        os.makedirs(model_dir, exist_ok=True)
        torch.save(checkpoint_data, model_path)
        print("Epoch {} train BCE: {:.4f}".format(epoch, train_loss.item()))

        # Logging
        model.eval()
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
    best_params = checkpoint["model_state_dict"]
    model.load_state_dict(best_params)

    # clean up not best models
    for i,model_path in enumerate(model_paths):
        if i != best_model_index:
            os.remove(model_path)

    return model, train_losses, val_losses


def test_loop(model, criterion, device, test_transform, test_dataset):
    loss = 0
    device = torch.device('cpu')   
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
            loss = loss + criterion(y_hat, y)
            labels.append(y)
            predictions.append(y_hat)

    loss = loss / test_dataset.snapshot_count
    loss = loss.item()

    print("Test BCE: {:.4f}".format(loss))
    return predictions, labels


# %%
number_sampels = 1000
dataset_name = "dataset_res8"
num_timesteps_in = 3
num_timesteps_out = 3

loader = TrajectoryDatasetLoader(f"../data/{dataset_name}.json", 
                                 num_timesteps_in,num_timesteps_out, 
                                 samples=number_sampels)
dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)

train_transform = MyTransform(train_set, 0,1)
test_transform = MyTransform(test_set, 0,1)

print("Number of train examples: ", train_set.snapshot_count)
print("Number of test examples: ", test_set.snapshot_count)

# %%
# experiment
filename = "experiment_top10.json"
weight_rebal = get_weight_rebal(train_set)
criterion = torch.nn.BCELoss(weight_rebal)
node_features = 7

linear_dims = [64] #[2**i for i in range(6,6+3)]
out_channels = [64] #[2**i for i in range(6,6+3)]
lrs = [0.1] #[10**(-i) for i in range(1, 1+3)]
ep = 1
rgcls = ["DCRNN"] #["DCRNN", "TGCN", "A3TGCN"]

create_json_file(filename)
settings = list(product(linear_dims, out_channels, lrs,rgcls))

print("start training")
for ld, oc, lr, rgcl in tqdm(settings):
    name = f"model_ld{ld}_oc{oc}_lr{lr}_rgcl{rgcl}_ds{dataset_name}_ep{ep}_ns{number_sampels}"
    print(name)
    
    config = {
        "linear_dim" : ld,
        "out_channels": oc,
        "learning_rate" : lr,
        "rgcl": rgcl,
        "dataset": dataset_name,
        "epochs": ep,
        "number_samples": number_sampels,
        "num_timesteps_in": num_timesteps_in,
        "num_timesteps_out": num_timesteps_out
        }
    
    device = "cpu"
    model = TemporalGNN(node_features, num_timesteps_in,num_timesteps_out,config)
    optimizer = torch.optim.Adam(model.parameters(), lr)

    # train_loop(name, model, criterion, optimizer,device, train_dataset, test_dataset, epochs)
    model,train_losses, val_losses = train_loop(name, 
                                                model, 
                                                criterion, 
                                                optimizer, 
                                                device,
                                                train_transform,
                                                test_transform,
                                                train_set, 
                                                test_set,
                                                config["epochs"])
    
    # test_loop(model, criterion, device, test_dataset, test_transform):
    predictions, labels = test_loop(model, 
                                    criterion, 
                                    device, 
                                    train_transform,
                                    train_set)
    
    # classification metrics
    labelss = torch.stack(labels)
    predictionss = torch.stack(predictions)
    best_threshold, best_fscore, auc_score = get_threshold_and_fscore(labels, predictions)
    bce = criterion(predictionss, labelss)
    
    result = {  
                "bce": bce.item(), 
                "best_threshold": best_threshold.item(),
                "fscore": best_fscore.item(),
                "train_losses": list(train_losses),
                "val_losses": list(val_losses)
                }
    
    entry = dict()
    dataset = {}
    entry[name] = config | result
    
    add_data_to_json(filename, entry)                
    save_path = os.path.join(os.curdir, "models", name+ ".pth")
    torch.save(model, save_path)


