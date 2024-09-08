# %%
import sys
import torch

sys.path.append('../')
sys.path.append('../../')

from torch_geometric_temporal.signal import temporal_signal_split
from tqdm import tqdm
from IPython.display import display

from master_thesis.utils.TemporalGNN import TemporalGNN, TemporalGNN_Batch
from master_thesis.utils.TrajectoryDatasetLoader import TrajectoryDatasetLoader
from master_thesis.utils.MyTransform import MyTransform
from master_thesis.utils.utils import  get_threshold_and_fscore
from master_thesis.utils.utils import create_json_file, add_data_to_json
from master_thesis.utils.utils import  visualize_y
from master_thesis.utils.utils import  train_loop

%load_ext autoreload
%autoreload 2

# %%
from master_thesis.utils.utils import load_from_checkpoint
from master_thesis.utils.utils import get_pos_weights

# %%
input_steps = 3
output_steps = 3

dataset_name = "dataset_res8"
node_features = 7

loader = TrajectoryDatasetLoader(f"../data/{dataset_name}.json", 
                                 input_steps,
                                 output_steps)
dataset = loader.get_dataset()
train_set, test_set = temporal_signal_split(dataset, train_ratio=0.8)
train_transform = MyTransform(train_set, 0,1)
test_transform = MyTransform(test_set, 0,1)

print("Number of train examples: ", train_set.snapshot_count)
print("Number of test examples: ", test_set.snapshot_count)

# %%
import numpy as np
batch_size = 32


# %%
#device = torch.device('cpu')
if torch.cuda.is_available():
    device = "cuda:0"

# %%
train_input = np.array(train_set.features) # (27399, 207, 2, 12)
train_target = np.array(train_set.targets) # (27399, 207, 12)
train_x_tensor = torch.from_numpy(train_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(device)  # (B, N, T)
train_dataset_new = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset_new, batch_size=batch_size, shuffle=True,drop_last=False)


# %%
test_input = np.array(test_set.features) # (, 207, 2, 12)
test_target = np.array(test_set.targets) # (, 207, 12)
test_x_tensor = torch.from_numpy(test_input).type(torch.FloatTensor).to(device)  # (B, N, F, T)
test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(device)  # (B, N, T)
test_dataset_new = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)
test_loader = torch.utils.data.DataLoader(test_dataset_new, batch_size=batch_size, shuffle=False,drop_last=False)



# %%
# https://www.kdnuggets.com/2021/09/imbalanced-classification-without-re-balancing-data.html
# https://discuss.pytorch.org/t/how-to-apply-a-weighted-bce-loss-to-an-imbalanced-dataset-what-will-the-weight-tensor-contain/56823
# https://datascience.stackexchange.com/questions/58735/weighted-binary-cross-entropy-loss-keras-implementation

# %%
rgcl = "A3TGCN2"
oc = 256
lr = 0.01
do = 0.5
eps = 2

config = {
                "rgcl": rgcl,
                "oc": oc,
                "lr" : lr,
                "do": do,
                "in": input_steps,
                "out": output_steps,
                "ds": dataset_name,
                "eps": eps,
                "nf": node_features,
                "batch_size" : 32,
                "device": device
            }

model = TemporalGNN_Batch(config)
optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"])
epochs = config["eps"]

weight_rebal = get_pos_weights(train_set).to(device)
criterion = torch.nn.BCELoss(weight_rebal).to(device)

# %%
from opacus import PrivacyEngine
from opacus.validators import ModuleValidator

errors = ModuleValidator.validate(model, strict=True)
errors


# %%
secure_mode = False 
privacy_engine = PrivacyEngine(secure_mode=secure_mode)  
# Privacy engine hyper-parameters 
MAX_GRAD_NORM = 1.5 
DELTA = 1e-5 
EPSILON = config["eps"]

EPOCHS = 20  
priv_model, priv_optimizer, priv_train_loader = privacy_engine.make_private_with_epsilon(
    module=model,     
    optimizer=optimizer,     
    data_loader=train_loader,     
    max_grad_norm=MAX_GRAD_NORM,     
    target_delta=DELTA,     
    target_epsilon=EPSILON,     
    epochs=EPOCHS )

# %%
priv_model = priv_model.to(device)

# %%
snapshot = next(iter(train_set))

edge_index = snapshot.edge_index.to(device)
edge_attr = snapshot.edge_attr.to(device)

# %%
from opacus.utils.batch_memory_manager import BatchMemoryManager
MAX_PHYSICAL_BATCH_SIZE = 32

# %%
priv_model.train()

# To device error to be fixed

with BatchMemoryManager(
        data_loader=priv_train_loader, 
        max_physical_batch_size=MAX_PHYSICAL_BATCH_SIZE, 
        optimizer=priv_optimizer
    ) as memory_safe_data_loader:
    
    for epoch in range(1):
        step = 0
        loss_list = []


        for encoder_inputs, labels in memory_safe_data_loader:
            optimizer.zero_grad()
            encoder_inputs =encoder_inputs.to(device)
            
            y_hat = priv_model(encoder_inputs, edge_index,edge_attr)
            loss = criterion(y_hat, labels)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            step= step+ 1
            loss_list.append(loss.item())
            if step % 100 == 0 :
                print(sum(loss_list)/len(loss_list))
        print("Epoch {} train RMSE: {:.4f}".format(epoch, sum(loss_list)/len(loss_list)))



