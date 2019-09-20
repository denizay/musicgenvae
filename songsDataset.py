import h5py
import numpy as np
from torch.utils.data import Dataset

class SongsDataset(Dataset):
    """Songs dataset."""

    def __init__(self):
        h5_file = h5py.File('songs.h5','r')
        self.songs = h5_file.get('dataset_1')
        self.len = self.songs.shape[0]

    def __len__(self):
    	return self.len

    def __getitem__(self, idx):

        return self.songs[idx]