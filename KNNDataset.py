import torch
from torch.utils.data import Dataset
import numpy as np

class KNNDataset(Dataset):
    def __init__(self, edge_index):
        self.edge_index = edge_index.T

    def __len__(self):
        return self.edge_index.shape[0]

    def __getitem__(self, idx):
        return self.edge_index[idx,:]

            
class CellDataset(Dataset):
    def __init__(self, x, knn, ordering):
        self.x = x
        self.knn = knn
        # Find the index of the gene with most non-zero expression across cells
        # Sum across cells (dim=0) to get number of cells with non-zero expression per gene
        # Then find the gene index with maximum value
        if ordering is not None:
            self.ordering = torch.tensor(ordering)
        else:
            self.ordering = None
        print(self.ordering)


    def __len__(self):
        return self.x.shape[1]

    def __getitem__(self, idx):
        return self.x[:,idx] , idx

    
class CustomDataset(Dataset):
    def __init__(self, x):
        self.data = x
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return torch.tensor(self.data[index])