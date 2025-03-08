
import os
import torch
from torch_geometric.data import Dataset


class MyDataset2(Dataset):
    def __init__(self, root, my_indices=None, net_adj=None):
        super(MyDataset2, self).__init__(root)
        self.my_indices = my_indices if my_indices is not None else range(len(self.processed_file_names))
        self.net_adj_set = set(zip(net_adj.row, net_adj.col))

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
        absolute_idx = self.my_indices[idx]
        data = torch.load(os.path.join(self.processed_dir, f'cell_{absolute_idx}.pt'))
        data.edge_index = data.edge_index.to(torch.int64)
        
        edge_index = data.edge_index
        edge_weight = data.edge_weight

        edge_weight = torch.log1p(edge_weight)
        mean_weight = edge_weight.mean().item()
        
        existing_edges = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
        new_edges = self.net_adj_set - existing_edges
        if new_edges:
            new_edges = torch.tensor(list(new_edges), dtype=torch.int64).T  
            new_weights = torch.tensor([mean_weight] * new_edges.size(1), dtype=torch.float32)

            edge_index = torch.cat([edge_index, new_edges], dim=1)
            edge_weight = torch.cat([edge_weight, new_weights])

        row, col = edge_index
        symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)

        symmetric_edge_weight = torch.cat([edge_weight, edge_weight])

        data.edge_index = symmetric_edge_index
        data.edge_weight = symmetric_edge_weight

        return data        
