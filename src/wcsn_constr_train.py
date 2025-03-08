import numpy as np
import pandas as pd
import os
import torch
from scipy.stats import norm
from scipy.sparse import coo_matrix
from torch_geometric.data import Data
import argparse
import time
import scanpy as sc

pathjoin = os.path.join

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr',type=str, default='data/pre_data/scRNAseq_datasets/Baron_Human.npz')
    parser.add_argument('-outdir', type=str, default='result/datasets/')
    parser.add_argument('-cuda','--cuda', type=bool, default=True)   
    parser.add_argument('-hvgs','--high_var_genes',type=int,default=2000)
    parser.add_argument('-ca', '--csn_alpha',type=float, default='0.01')
    parser.add_argument('-nsp', '--n_splits',type=int,default=5)
    return parser


def csn_constr_train_without(args):
    expr_npz = args.expr
    save_folder = args.outdir
    csn_alpha = args.csn_alpha

    HVGs_num = args.high_var_genes
    
    n_splits=args.n_splits

    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    seq_folder = pathjoin(save_folder, base_filename)  
    csn_data_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}")
    
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']

    logExpr0 = get_logExpr3(expr_npz) 
    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
   
    for k in range(n_splits):
        k_fold = k + 1
        print("train k_fold: ", k_fold)
        train_index = seq_dict[f'train_index_{k_fold}'] 
        label_train = label[train_index]  

        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)
        
        logExpr0_train = logExpr0[np.ix_(train_index, filtered_genes_index)] 
        cell_train_folder = os.path.join(csn_data_folder, f"train_f{k_fold}")
        com_csn_constr_and_save(logExpr0_train,label_train, alpha=csn_alpha, save_folder=cell_train_folder)
        
        print(f"train fold{k_fold} csn is completed!")



def get_logExpr3(expr_npz):
    data = np.load(expr_npz, allow_pickle=True)

    countExpr = data['count'] 
    print("raw (cells, genes): ", countExpr.shape)

    row_sums = countExpr.sum(axis=1, keepdims=True)
    normalized_data = 1e6 * countExpr / row_sums
    normalized_data = normalized_data.astype(np.float32)
    logExpr0 = np.log1p(normalized_data) 

    return logExpr0



def com_csn_constr_and_save(logExpr0, label, c=None, alpha=0.01, boxsize=0.1, save_folder='data/processed_csn'):
    data = logExpr0.transpose()
    n1, n2 = data.shape 
    print("gene num, n1:", n1)
    print("cell num, n2:", n2)

    if c is None:
        c = torch.arange(n2).cuda()
    else:
        c = torch.arange(c).cuda()

    os.makedirs(save_folder, exist_ok=True)
    processed_dir = os.path.join(save_folder, 'processed')
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)

    upper = np.zeros((n1, n2))
    lower = np.zeros((n1, n2))
    for i in range(n1):
        s1 = np.sort(data[i, :])
        s2 = np.argsort(data[i, :])
        n3 = n2 - np.sum(np.sign(s1))
        h = round(boxsize / 2 * np.sum(np.sign(s1)))
        k = 0
        while k < n2:
            s = 0
            while k + s + 1 < n2 and s1[k + s + 1] == s1[k]:
                s += 1
            if s >= h:
                upper[i, s2[k:k+s+1]] = data[i, s2[k]]
                lower[i, s2[k:k+s+1]] = data[i, s2[k]]
            else:
                upper[i, s2[k:k+s+1]] = data[i, s2[min(n2-1, k+s+h)]]
                lower[i, s2[k:k+s+1]] = data[i, s2[max(int(n3*(n3>h)), k-h)]]

            k = k + s + 1

    data = torch.from_numpy(data).cuda()
    upper = torch.from_numpy(upper).cuda()
    lower = torch.from_numpy(lower).cuda()

    p = -norm.ppf(alpha) 
    print(p)

    for k in c:
        print(f"cell {k}")
        B = ((data <= upper[:, k].unsqueeze(1)) & (data >= lower[:, k].unsqueeze(1)) & (data[:, k].unsqueeze(1) > 0)).float()
        a = torch.sum(B, dim=1)
        score = (B @ B.T * n2 - torch.outer(a, a)) / torch.sqrt(torch.outer(a, a) * torch.outer(n2 - a, n2 - a) / (n2 - 1) + torch.finfo(torch.float32).eps)
        score[score < p] = 0  
        
        score = torch.triu(score, diagonal=1)

        indices = torch.nonzero(score, as_tuple=False).t().contiguous()
        values = score[indices[0], indices[1]]
        sparse_score = torch.sparse_coo_tensor(indices, values, score.shape).coalesce().to(torch.float32)
        print("score: ", len(values))

        sparse_score = sparse_score.coalesce()      
        sparse_score = sparse_score.cpu()       

        save_data2(logExpr0[k], sparse_score, label[k], k, processed_dir)
        
    print('csn construction is completed!')




def save_data2(logExpr_i, com_csn, label_i, i, save_folder):
    x = logExpr_i   
    x = x.reshape(-1, 1)  
    x = x.astype(np.float32)
    x = torch.from_numpy(x)

    if isinstance(com_csn, coo_matrix):
        indices = np.array([com_csn.row, com_csn.col], dtype=np.int32)
        edge_index_torch = torch.tensor(indices, dtype=torch.int32)

        edge_weight = torch.tensor(com_csn.data, dtype=torch.float32)
        
    elif isinstance(com_csn, torch.sparse.Tensor):
        graph = com_csn.coalesce()
        edge_index_torch = graph.indices().to(torch.int32)
        edge_weight = graph.values().to(torch.float32)
    
    else:
        raise TypeError("Unsupported com_csn type: {}".format(type(com_csn)))

    y = torch.tensor([label_i], dtype=torch.long)
    cell_data = Data(x=x, edge_index=edge_index_torch, edge_weight=edge_weight, y=y)
    
    save_path = os.path.join(save_folder, f'cell_{i}.pt')
    torch.save(cell_data, save_path)


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)
    
    csn_constr_train_without(args)
    
    end_time = time.time()
    print(f"Run time: {end_time - start_time} s")