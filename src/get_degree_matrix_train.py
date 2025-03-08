import numpy as np
import os
import torch
import argparse
import time

pathjoin = os.path.join

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr',type=str, default='data/pre_data/scRNAseq_datasets/Baron_Human.npz')  
    parser.add_argument('-outdir', type=str, default='result/datasets/') 
    parser.add_argument('-hvgs','--high_var_genes',type=int,default=2000)
    parser.add_argument('-ca', '--csn_alpha',type=float, default='0.01')

    return parser


def get_matrix(args):
    expr_npz = args.expr
    save_folder = args.outdir
    csn_alpha = args.csn_alpha
    HVGs_num = args.high_var_genes
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      

    seq_folder = pathjoin(save_folder, base_filename)

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    print("cell type: ", str_labels)

    cur_label = 0
    matrix_dict = {}
    matrix_dict['str_labels'] = str_labels

    for cell_type in str_labels:
        print("cur_label: ", cur_label)
        print("cell_type: ", cell_type)

        degree_matrices_for_folds = {} 
        
        for k in range(5):
            k_fold = k + 1
            train_index = seq_dict[f'train_index_{k_fold}']
            label_train = label[train_index]
            cur_label_idxs = np.where(label_train == cur_label)[0].tolist()  
            cell_train_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}", f"train_f{k_fold}", 'processed')
            
            degree_matrix = []

            for idx in cur_label_idxs:
                data = torch.load(os.path.join(cell_train_folder, f'cell_{idx}.pt'))
                edge_index = data.edge_index
                edge_weight = data.edge_weight
                row, col = edge_index
                symmetric_edge_index = torch.cat([edge_index, torch.stack([col, row])], dim=1)
                degrees = torch.bincount(symmetric_edge_index[0])
                if degrees.size(0) < 2000:
                    degrees = torch.cat([degrees, torch.zeros(2000 - degrees.size(0))])

                degree_matrix.append(degrees.numpy())

            degree_matrix = np.array(degree_matrix).T
            print(degree_matrix.shape)
            degree_matrices_for_folds[f'CV_{k_fold}'] = degree_matrix  

        matrix_dict[str(cur_label)] = degree_matrices_for_folds
        cur_label += 1

    degree_file = pathjoin(seq_folder, f'degree_matrix_train_{base_filename}_a{csn_alpha}_hvgs{HVGs_num}.npz')

    print(f"Matrix dict structure before saving: {type(matrix_dict)}")
    for key in matrix_dict:
        print(f"Key: {key}, Type: {type(matrix_dict[key])}")
        if isinstance(matrix_dict[key], dict):  
            for inner_key in matrix_dict[key]:
                print(f" Inner Key: {inner_key}, Type: {type(matrix_dict[key][inner_key])}")


    np.savez(degree_file, **matrix_dict)



if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    get_matrix(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")