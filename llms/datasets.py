import h5py
import torch
from torch.utils.data import Dataset

class TokensDataset(Dataset):
    def __init__(self, tokens_filename:str, partition:str='train', block_size:int=1024):
        super(TokensDataset, self).__init__()

        with h5py.File(tokens_filename, 'r') as dataset_filename:
            if partition == 'train':
                self._examples = dataset_filename[partition][:]
            else: # partition == 'test'
                self._examples = dataset_filename[partition][:]

    def __len__(self):
        return len(self._examples)
    
    def __getitem__(self, index):
        return torch.tensor(self._examples[index])
