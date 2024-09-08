import torch
import pandas as pd
import numpy as np


class PersonalTrajectoryTestDataset(torch.utils.data.Dataset):

    # Basic Instantiation
    def __init__(self, json_file, transform=None):
        self.transform = transform
        self.NONE_HEX = "NONE_HEX"
        self.NONE_PURPOSE = "NONE_PURPOSE"
        self.NONE_TIME = 0

        # set correctly by __preprocessing__
        self.max_trajectory_length = 0
        
        self.df = pd.read_json(json_file)
        self.hex_to_id, self.purpose_to_id = self.__build_mappings__(self.df)

        self.personal_datas, self.source_trajectories, self.target_trajectories = self.__preprocessing__(self.df)


    
    def encode_trajectorie_history(self, x):
        # add padding
        x = x +(self.max_trajectory_length - len(x))* [[self.NONE_HEX, self.NONE_PURPOSE, self.NONE_TIME]]
        result = []
        #encode
        for test  in x:
            hex, purpose, time = test
            hex_id = self.hex_to_id[hex]
            purpose_id = self.purpose_to_id[purpose]
            #time_day_sin = sin_transformer(time,365)
            #time_day_cos = cos_transformer(time,365)    
            result.append([hex_id, purpose_id, time])

        return result

    def __build_mappings__(self, df):
        hexes ={self.NONE_HEX}
        purposes = {self.NONE_PURPOSE}

        for entry in df["source_trajectories"]:
            for hex, purpose, time in entry:
                hexes = hexes.union({hex})
                purposes = purposes.union({purpose})

        for entry in df["target_trajectories"]:
            for hex, purpose, time in entry:
                hexes = hexes.union({hex})
                purposes = purposes.union({purpose})

        hexes = hexes.difference({self.NONE_HEX})
        purposes = purposes.difference({self.NONE_PURPOSE})

        hex_to_id = {hex: id+1 for id, hex in enumerate(hexes)}
        hex_to_id[self.NONE_HEX] = 0

        purpose_to_id = {purpose: id+1 for id, purpose in enumerate(purposes)}
        purpose_to_id[self.NONE_PURPOSE] = 0

        return hex_to_id, purpose_to_id

    
    def __preprocessing__(self, df):
        self.max_trajectory_length = max(max(df["source_trajectories"].apply(len)),max(df["target_trajectories"].apply(len)))

        # encode trajectories
        df["source_trajectories"] = df["source_trajectories"].apply(self.encode_trajectorie_history)
        df["target_trajectories"] = df["target_trajectories"].apply(self.encode_trajectorie_history)

        # map them to numpy
        source_trajectories_np = np.array(df["source_trajectories"].to_list(), dtype=float)
        target_trajectories_np = np.array(df["target_trajectories"].to_list(), dtype=float)
        personal_datas_np = np.array(df["personal_datas"].to_list(), dtype=float)
        return personal_datas_np, source_trajectories_np, target_trajectories_np


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        sample = {'personal_data': self.personal_datas[idx], 
                  'source_trajectory': self.source_trajectories[idx], 
                  'target_trajectory': self.target_trajectories[idx]}

        if self.transform:
            sample = self.transform(sample)

        return sample
    
if __name__ == "__main__":
    print("Hello, World!")