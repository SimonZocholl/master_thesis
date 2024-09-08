import json
import numpy as np
from torch_geometric_temporal import StaticGraphTemporalSignal

class TrajectoryDatasetLoader():
    def __init__(self, data_path: str, num_timesteps_in : int=3, num_timesteps_out: int=3, x_features: list[str] = ["hexagon_work","hexagon_leisure","hexagon_errand", "user_work","user_errand","user_leisure", "visited"], y_feature : str = "visited", samples: int = -1):
        #self._read_web_data(data_path)
        
        # graph setup
        self._dataset = self._read_web_data(data_path)
        self._edges = self._get_edges()
        self._edge_weights = self._get_edge_weights()
        self.samples = samples
        self.num_timesteps_in = num_timesteps_in
        self.num_timesteps_out = num_timesteps_out

        # lookup tables and values
        self.resolution = self._dataset["resolution"]
        self.features = self._dataset["features"]
        self.x_features = x_features
        self.y_feature = y_feature

        self.hex_names_to_ids, self.ids_to_hex_names = self._get_dict_and_reverse_dict("hex_names_to_ids")
        self.activity_to_ids, self.ids_to_activity = self._get_dict_and_reverse_dict("activity_to_ids")
        self.user_id_to_ids, self.ids_to_user_id = self._get_dict_and_reverse_dict("user_id_to_ids")
        self.features_to_ids, self.ids_to_features = self._get_dict_and_reverse_dict("features_to_ids")
        self.sample_to_user_id = dict()

        self.x, self.y = self._generate_task()
        self.dataset = self.get_dataset()


    def _read_web_data(self,data_path):
        #url = url = "https://drive.usercontent.google.com/download?id=1gOnORRGuQTj9bywhy0k3nHtPaG759REH&export=download&authuser=0&confirm=t&uuid=be2c6aed-1a9e-4212-9bee-8bdd7eb8a369&at=APZUnTW1Lah6_JffvhIA4K05Lass%3A1709722418583"
        #self._dataset = json.loads(urllib.request.urlopen(url).read())
        with open(data_path) as f:
            _dataset =json.load(f)
            
        return _dataset
    
    def _get_dict_and_reverse_dict(self, dict_name):
        dct = self._dataset[dict_name]
        rev_dict = {v:k for k,v in dct.items()}
        return dct, rev_dict     
    

    def _get_edges(self):
        return np.array(self._dataset["edges"]).T

    
    def _get_edge_weights(self):
        # np.ones(self._edges.shape[1])
        #edge_weights = np.array(self._dataset["edge_usages"])+1
        #edge_weights_normalized = edge_weights / np.sum(edge_weights)
        
        return np.ones(self._edges.shape[1])
    
    def _generate_task(self):
        # (examples, nodes, features) -> (nodes, features, examples)
        Xt = np.transpose(self._dataset["X"], (1, 2, 0))
        self.x_ids = [self.features_to_ids[x_feature] for x_feature in self.x_features]
        self.y_id = self.features_to_ids[self.y_feature]

        x,y = [], []
        total_timesteps = self.num_timesteps_in + self.num_timesteps_out
        samples = 0
        user_id_index = self.features_to_ids["user_id"]
        i = 0
        while i < Xt.shape[2]-(self.num_timesteps_in + self.num_timesteps_out) and (samples < self.samples or self.samples ==-1):
            j = i + total_timesteps
            
            if Xt[0,user_id_index,i] == Xt[0,user_id_index,j]:
                self.sample_to_user_id[samples] = Xt[0,user_id_index,i]

                x.append((Xt[:, self.x_ids, i : i + self.num_timesteps_in]))
                y.append((Xt[:, self.y_id, i + self.num_timesteps_in : j]))
                samples += 1
                i = j
            else:
                i = i +1

        x, y = np.array(x), np.array(y)
        return x,y

    '''
    def _generate_task(self):
        # (examples, nodes, features) -> (nodes, features, examples)
        Xt = np.transpose(self._dataset["X"], (1, 2, 0))
        self.x_ids = [self.features_to_ids[x_feature] for x_feature in self.x_features]
        self.y_id = self.features_to_ids[self.y_feature]

        total_timesteps = self.num_timesteps_in + self.num_timesteps_out
        indices = [(i, i + (total_timesteps))for i in range(0,Xt.shape[2]-(total_timesteps), total_timesteps)]

        x,y = [], []

        samples = 0
        user_id_index = self.features_to_ids["user_id"]
        for i,j in indices:
            # same user from start to end
            if (samples < self.samples or self.samples ==-1) and Xt[0,user_id_index,i] == Xt[0,user_id_index,j]:
                self.sample_to_user_id[samples] = Xt[0,user_id_index,i]
                x.append((Xt[:, self.x_ids, i : i + self.num_timesteps_in]))
                y.append((Xt[:, self.y_id, i + self.num_timesteps_in : j]))
                samples += 1

        x, y = np.array(x), np.array(y)
        return x,y
    '''
        
    def get_dataset(self) -> StaticGraphTemporalSignal:
        
        dataset = StaticGraphTemporalSignal(
            self._edges, self._edge_weights, self.x, self.y
        )
        return dataset