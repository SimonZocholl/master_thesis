import torch
from torch_geometric_temporal.nn.recurrent import A3TGCN, DCRNN, TGCN, GCLSTM
import torch.nn.functional as F
from torch_geometric_temporal.nn.recurrent import A3TGCN2, TGCN2


class TemporalGNN(torch.nn.Module):
    def __init__(self, config):
        super(TemporalGNN, self).__init__()
        self.config = config
        self.num_timesteps_in = config["in"]
        self.num_timesteps_out = config["out"]
        self.node_features = config["nf"]

        self.out_channels = config["oc"]
        self.do = config["do"]
        self.ld = config["ld"]

        # checked, weight
        if self.config["rgcl"] ==  "A3TGCN":
            self.tgnn = A3TGCN(self.node_features,self.out_channels,self.num_timesteps_in)
        # checked, weight
        elif self.config["rgcl"] == "DCRNN": #
            self.tgnn =  DCRNN(self.node_features,self.out_channels,1)
        # checked, weight
        elif self.config["rgcl"] == "TGCN": #
            self.tgnn = TGCN(self.node_features,self.out_channels)
        elif self.config["rgcl"]  == "GCLSTM":
             self.tgnn = GCLSTM(self.node_features,self.out_channels, 1)
        else:
            print("error")
        
        # Equals single-shot prediction
        self.linear1 = torch.nn.Linear(self.out_channels, self.ld)
        self.linear2 = torch.nn.Linear(self.ld, self.num_timesteps_out)
        self.dropout = torch.nn.Dropout(self.do)

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = None
        if self.config["rgcl"] in ["DCRNN", "TGCN"]:            
            h = self.tgnn(x[:,:,0], edge_index, edge_attr)
            for i in range(1,x.shape[-1]):
                h = self.tgnn(x[:,:,i], edge_index, edge_attr, H=h)
        elif self.config["rgcl"] == "GCLSTM":
            h, c = self.tgnn(x[:,:,0], edge_index, edge_attr)
            for i in range(1,x.shape[-1]):
                h, c = self.tgnn(x[:,:,i], edge_index, edge_attr, h, c)
        else:
            h = self.tgnn(x, edge_index, edge_attr)
        
                
        h = F.relu(h)
        h = self.dropout(h)
        h = F.relu(self.linear1(h))
        h = self.linear2(h)
        return h
    

class TemporalGNN_Batch(torch.nn.Module):
    def __init__(self, config):
        super(TemporalGNN_Batch, self).__init__()
        self.config = config
        self.num_timesteps_in = config["in"]
        self.num_timesteps_out = config["out"]
        self.node_features = config["nf"]
        self.out_channels = config["oc"]
        self.do = config["do"]

        # Attention Temporal Graph Convolutional Cell
        # checked, weight
        if config["rgcl"] ==  "A3TGCN2":# 2020
            self.tgnn = A3TGCN2(self.node_features,self.out_channels,self.num_timesteps_in, config["batch_size"])
        # checked, weight
        elif config["rgcl"] == "TGCN2": # 2018
            self.tgnn = TGCN2(self.node_features,self.out_channels,config["batch_size"])
        else:
            print("error")
        
        # Equals single-shot prediction
        self.linear = torch.nn.Linear(self.out_channels, self.num_timesteps_out)
        self.dropout = torch.nn.Dropout(config["do"])

    def forward(self, x, edge_index, edge_attr):
        """
        x = Node features for T time steps
        edge_index = Graph edge indices
        """
        h = None
        if self.config["rgcl"] == "TGCN2":            
            h = self.tgnn(x[:,:,0], edge_index, edge_attr)
            for i in range(1,x.shape[-1]):
                h = self.tgnn(x[:,:,i], edge_index, edge_attr, H=h)
        else:
            h = self.tgnn(x, edge_index, edge_attr)
        
        h = F.relu(h)
        h = self.dropout(h)
        h = self.linear(h)
        return h
