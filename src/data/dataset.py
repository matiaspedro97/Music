import torch

from src.data.parser import *
from src.utils.constants import *


class DataLoader:
    def __init__(self, parse_func: str, dset_path: str) -> None:
        self.parse_func = eval(parse_func)

        if len(dset_path.split(os.sep)) > 1:
            self.dset_path = dset_path 
        else:
            self.dset_path = eval(dset_path)

    def transform(self):
        df_parse = self.parse_func(self.dset_path)
        return df_parse
    

class CustomAudioDataset:
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]).type(torch.float) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx]).type(torch.LongTensor)
        return item

    def __len__(self):
        return len(self.labels)