import numpy as np
import pandas as pd
import os
import torch
from scipy.stats import norm
from scipy.sparse import csr_matrix, coo_matrix, lil_matrix
from sklearn.model_selection import StratifiedKFold
from collections import Counter 
from sklearn.model_selection import StratifiedShuffleSplit
import argparse
import time
import scanpy as sc

import torch.nn.functional as F
from sklearn.metrics import precision_score,f1_score, accuracy_score
from torch_geometric.data import Dataset
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from scipy.sparse import load_npz

from model1 import CSGNet
from datasets import MyDataset

pathjoin = os.path.join


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr',type=str, default='data/pre_data/scRNAseq_datasets2/Baron_Human.npz')  # 输入文件，这里是npz文件
    parser.add_argument('-net','--net',type=str, default='data/pre_data/network/HumanNet-GSP.tsv')   # 基因相互作用网络，使用黄金标准
    parser.add_argument('-outdir', type=str, default='data/scNet_result3/')
    parser.add_argument('-cuda','--cuda', type=bool, default=True)
    parser.add_argument('-bs','--batch_size',type=int,default=32)
    parser.add_argument('-epoch' ,type=int,default=30)
    parser.add_argument('-ispart', type=bool, default=False)
    parser.add_argument('-p', '--part_data',type=float, default='1')
    
    # 基因筛选
    parser.add_argument('-hvgs','--high_var_genes',type=int,default=2000)
    parser.add_argument('-hdgs','--high_digree_genes',type=int,default=0)
    parser.add_argument('-hvdgs','--high_var_digree_genes',type=int,default=0)

    parser.add_argument('-ca', '--csn_alpha',type=float, default='0.01')
    parser.add_argument('-ed', '--exist_data', type=bool, default=False)
    parser.add_argument('-addname',type=str, default="")
    parser.add_argument('-wf', '--weight_flag', type=str, default='mean')
    parser.add_argument('-other',type=str, default="")
    parser.add_argument('-nsp', '--n_splits',type=int,default=5)
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64],help='模型参数list')
    parser.add_argument('-mc','--mid_channel',type=int,default=16)
    parser.add_argument('-gcdim1','--global_conv1_dim', type=int, default=12)
    parser.add_argument('-gcdim2','--global_conv2_dim', type=int, default=4)

    parser.add_argument('-nw','--num_workers', type=int, default=4)
    
    return parser




def classify_test(args):
    expr_npz = args.expr
    save_folder = args.outdir
    csn_alpha = args.csn_alpha

    HVGs_num = args.high_var_genes
    
    addname = args.addname 
    n_splits=args.n_splits

    cuda_flag = args.cuda
    batch_size = args.batch_size

    num_workers = args.num_workers

    # 处理文件名称，例如：得到base_filename为Baron_Human
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]      


    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag  else 'cpu')

    # 读取seq_dict 五折划分文件
    seq_folder = pathjoin(save_folder, base_filename)
    # models_folder = pathjoin(save_folder, 'models')
    models_folder = pathjoin(seq_folder, 'wcsn_models')


    os.makedirs(pathjoin(save_folder, 'wcsn_preds'),exist_ok=True)
    preds_folder = pathjoin(save_folder, 'wcsn_preds')
    csn_data_folder = pathjoin(seq_folder, f"{base_filename}_a{csn_alpha}_hvgs{HVGs_num}")

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    # genes_id = seq_dict['gene_id']
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    barcodes = seq_dict['barcode']

    # gene_num = len(genes_symbol)
    class_num = len(np.unique(str_labels))

    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    # 得到每一折的基因数量：
    # genes_num_all = get_gene_num(all_filtered_genes_file)
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)

    # 用于保存每个细胞是每个细胞类型的概率
    pred_probability = []
    # 用于保存每个细胞的预测细胞类型以及真实细胞类型
    cell_type_all = []
    # 用于保存每个细胞最后的编码
    cell_embedding= []

    n_splits = args.n_splits
    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold: ", k_fold)
        test_index = seq_dict[f'test_index_{k_fold}']
        barcodes_test = barcodes[test_index]
        label_test = label[test_index]
        cell_test_folder = os.path.join(csn_data_folder, f"test_f{k_fold}")
        
        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)

        # 训练集和测试集的构建
        test_dataset = MyDataset(root=cell_test_folder)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        # 加载模型的状态字典
        model_file = pathjoin(models_folder, f'{base_filename}_a{csn_alpha}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')
        # model.load_state_dict(torch.load(model_file))
        # 确保文件存在
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        # 加载整个模型
        model = torch.load(model_file)
        # 将模型移动到设备上
        model.to(device)



        test_acc, test_f1, curr_y_out, curr_y_pred, cell_latent = test(model, test_loader, device, predicts=True, latent=True) 
        print('Acc: %.03f, F1: %.03f' %(test_acc, test_f1))   
        cell_latent = pd.DataFrame(cell_latent, index=barcodes_test) 
        cell_embedding.append(cell_latent)   
        curr_y_out = pd.DataFrame(curr_y_out, index=barcodes_test) 
        pred_probability.append(curr_y_out)
        sub_test_cell_type = pd.DataFrame(index=barcodes_test)
        sub_test_cell_type["pred_cell_type"] = curr_y_pred
        sub_test_cell_type["true_cell_type"] = label_test
        cell_type_all.append(sub_test_cell_type)     


    # 得到所有test细胞，也就是所有折的预测结果
    # 循环结束后，将所有DataFrame组合在一起
    pred_probability_df = pd.concat(pred_probability, ignore_index=False)  # 保留原始索引
    cell_type_all_df = pd.concat(cell_type_all, ignore_index=False) # 保留原始索引
    cell_embedding_df = pd.concat(cell_embedding, ignore_index=False) # 保留原始索引 

    # 计算acc, f1
    label_true = cell_type_all_df['true_cell_type'].to_numpy()
    label_pred = cell_type_all_df["pred_cell_type"].to_numpy()
    test_acc = accuracy_score(label_true, label_pred)
    # acc = precision_score(y_true, y_pred, average='macro')
    test_f1 = f1_score(label_true, label_pred, average='macro')
    test_f1_all = f1_score(label_true, label_pred, average=None)
    print('Final Acc: %.03f, F1: %.03f'%(test_acc, test_f1))   


    pred_save_file = pathjoin(preds_folder, f'{base_filename}_a{csn_alpha}_hvgs{HVGs_num}{addname}_prediction.h5')
    cell_type_all_df.to_hdf(pred_save_file, key='cell_type', mode='a')
    pred_probability_df.to_hdf(pred_save_file, key='pred_prob', mode='a')
    cell_embedding_df.to_hdf(pred_save_file, key='embedding', mode='a')


    f1_all = pd.DataFrame(index=str_labels)
    f1_all['F1'] = test_f1_all
    # 将 test_acc 和 test_f1 添加到 f1_all 中的最后两行
    f1_all.loc['acc'] = [test_acc]
    f1_all.loc['macro_f1'] = [test_f1]
    f1_file = pathjoin(preds_folder, f'{base_filename}_a{csn_alpha}_hvgs{HVGs_num}{addname}_F1.csv')
    print(f1_all)
    f1_all.to_csv(f1_file, index=True)   



def test(model, loader, device, predicts=False, latent=False):
    model.eval()
    y_pred = []
    y_true = []
    y_out = []
    cell_latent = []
    for data in loader:
        data = data.to(device)
        if latent:
            latent_varaible, y_output = model(data, get_latent_varaible=True)
            latent_var = latent_varaible.cpu().data.numpy()
            cell_latent.append(latent_var)
        else:
            y_output = model(data)
        # pred = y_output.cpu().data.numpy()
        y_softmax = F.softmax(y_output, dim=1).cpu().detach().numpy() # Convert to probabilities
        y_out.extend(y_softmax)
        
        pred = y_output.argmax(dim=1).cpu().numpy()
        y = data.y.cpu().data.numpy()
        y_pred.extend(pred) 
        y_true.extend(y) #(64,)

    acc = accuracy_score(y_true, y_pred)
    # acc = precision_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    if predicts:
        if latent:
            # 在循环结束后，将所有的 latent_var 垂直堆叠成一个大的 NumPy 数组
            cell_latent = np.vstack(cell_latent)
            return acc, f1, y_out, y_pred, cell_latent
        else:
            return acc, f1, y_out, y_pred
    else:
        if latent:
            cell_latent = np.vstack(cell_latent)
            return acc, f1, cell_latent
        else:
            return acc, f1



def get_gene_num(all_filtered_genes_file):
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
    # 获取每一行的长度
    genes_num = [len(row) for row in all_filtered_genes_array]
    return genes_num


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    classify_test(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")

