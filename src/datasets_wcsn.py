import os
import torch
from torch_geometric.data import Dataset

class MyDataset(Dataset):

    def __init__(self, root, my_indices=None):
        super(MyDataset, self).__init__(root)
        # If my_indices is provided, use it; otherwise, use all indices from the processed data
        self.my_indices = my_indices if my_indices is not None else range(len(self.processed_file_names))
    
    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        cell_files = os.listdir(self.processed_dir)
        return cell_files

    def download(self):
        pass

    def process(self):
        pass

    def len(self):
        return len(self.my_indices)

    def get(self, idx):
        """
        Loads and returns the processed data for a given index.
        
        Args:
            idx (int): The index of the data sample to retrieve.

        Returns:
            torch_geometric.data.Data: The data object containing the node features, edge indices, and edge weights. (WCSN)
        """
        # Retrieve the absolute index from the list of indices
        absolute_idx = self.my_indices[idx]
        # Load the preprocessed data from the disk
        data = torch.load(os.path.join(self.processed_dir, f'cell_{absolute_idx}.pt'))
        data.edge_index = data.edge_index.to(torch.int64)

        # Get the current edge indices and edge weights
        edge_index = data.edge_index
        edge_weight = data.edge_weight
        
        # Generate symmetric edge indices by adding reverse edges (i.e., (i, j) and (j, i))
        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
        
        # Generate symmetric edge weights by duplicating the weights
        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])
        
        # Update the data with the symmetric edge indices and weights
        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight
        
        return data
