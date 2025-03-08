import numpy as np
import pandas as pd
import os
import torch
import argparse
import time
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from scipy.sparse import load_npz

from datasets_ewcsn import MyDataset2

pathjoin = os.path.join


def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-net', '--net', type=str, default='HumanNet-GSP')   # Gene interaction network (gold standard)
    parser.add_argument('-outdir', type=str, default='result/')  # Output directory
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)  # Use CUDA (GPU) for computation
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)  # Number of high variance genes
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)  # Significance level for WCSN construction
    parser.add_argument('-addname', type=str, default="") 
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64], help='Model parameter list') 
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)  # Mid channel dimension
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)  # First global convolution dimension
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)  # Second global convolution dimension
    return parser


def classify_test(args):
    expr_npz = args.expr
    net_filename = args.net
    save_folder = args.outdir
    csn_alpha = args.csn_alpha
    HVGs_num = args.high_var_genes
    addname = args.addname 
    n_splits = args.n_splits
    cuda_flag = args.cuda
    batch_size = args.batch_size

    # Process file names
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag else 'cpu')

    # Define directory paths for data and models
    seq_folder = pathjoin('result/datasets/', base_filename)

    models_folder = pathjoin(save_folder, 'ewcsn_models')
    os.makedirs(pathjoin(save_folder, 'ewcsn_preds'), exist_ok=True)
    preds_folder = pathjoin(save_folder, 'ewcsn_preds')

    csn_data_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}")

    net_adj_file = pathjoin(seq_folder, f'{base_filename}_{net_filename}_net_adj.npz')    
    net_adj = load_npz(net_adj_file)

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']
    barcodes = seq_dict['barcode']

    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)

    # Initialize lists to store prediction results and embeddings
    pred_probability = []
    cell_type_all = []
    cell_embedding = []

    # Loop over the splits for cross-validation
    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold:", k_fold)
        test_index = seq_dict[f'test_index_{k_fold}']
        barcodes_test = barcodes[test_index]
        label_test = label[test_index]
        cell_test_folder = os.path.join(csn_data_folder, f"test_f{k_fold}")

        filtered_genes_index = all_filtered_genes_array[k]
        filtered_genes_index = filtered_genes_index.astype(int)

        # Subset the network adjacency matrix based on filtered genes
        cur_net_adj = net_adj.tocsr()[filtered_genes_index, :][:, filtered_genes_index].tocoo()

        # Create test dataset and loader, get EWCSN
        test_dataset = MyDataset2(root=cell_test_folder, net_adj=cur_net_adj)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

        # Load the model
        model_file = pathjoin(models_folder, f'{base_filename}_{net_filename}_a{csn_alpha}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')
        if not os.path.isfile(model_file):
            raise FileNotFoundError(f"Model file not found: {model_file}")

        model = torch.load(model_file)
        model.to(device)

        # Perform testing and get results
        test_acc, test_f1, curr_y_out, curr_y_pred, cell_latent = test(model, test_loader, device, predicts=True, latent=True) 
        print('Acc: %.03f, F1: %.03f' % (test_acc, test_f1))   

        # Store the results
        cell_latent = pd.DataFrame(cell_latent, index=barcodes_test) 
        cell_embedding.append(cell_latent)
        curr_y_out = pd.DataFrame(curr_y_out, index=barcodes_test) 
        pred_probability.append(curr_y_out)
        
        sub_test_cell_type = pd.DataFrame(index=barcodes_test)
        sub_test_cell_type["pred_cell_type"] = curr_y_pred
        sub_test_cell_type["true_cell_type"] = label_test
        cell_type_all.append(sub_test_cell_type)     

    # Concatenate all predictions and embeddings
    pred_probability_df = pd.concat(pred_probability, ignore_index=False)
    cell_type_all_df = pd.concat(cell_type_all, ignore_index=False)
    cell_embedding_df = pd.concat(cell_embedding, ignore_index=False)

    # Calculate accuracy and F1 score
    label_true = cell_type_all_df['true_cell_type'].to_numpy()
    label_pred = cell_type_all_df["pred_cell_type"].to_numpy()
    test_acc = accuracy_score(label_true, label_pred)
    test_f1 = f1_score(label_true, label_pred, average='macro')
    test_f1_all = f1_score(label_true, label_pred, average=None)
    print('Final Acc: %.03f, F1: %.03f' % (test_acc, test_f1))

    # Save predictions and F1 scores
    pred_save_file = pathjoin(preds_folder, f'{base_filename}_{net_filename}_a{csn_alpha}_hvgs{HVGs_num}{addname}_prediction.h5')
    cell_type_all_df.to_hdf(pred_save_file, key='cell_type', mode='a')
    pred_probability_df.to_hdf(pred_save_file, key='pred_prob', mode='a')
    cell_embedding_df.to_hdf(pred_save_file, key='embedding', mode='a')

    # Save F1 score results
    f1_all = pd.DataFrame(index=str_labels)
    f1_all['F1'] = test_f1_all
    f1_all.loc['acc'] = [test_acc]
    f1_all.loc['macro_f1'] = [test_f1]
    f1_file = pathjoin(preds_folder, f'{base_filename}_{net_filename}_a{csn_alpha}_hvgs{HVGs_num}{addname}_F1.csv')
    print(f1_all)
    f1_all.to_csv(f1_file, index=True)


def test(model, loader, device, predicts=False, latent=False):
    model.eval()
    y_pred = []
    y_true = []
    y_out = []
    cell_latent = []
    
    # Loop over the test data
    for data in loader:
        data = data.to(device)
        if latent:
            latent_varaible, y_output = model(data, get_latent_varaible=True)
            latent_var = latent_varaible.cpu().data.numpy()
            cell_latent.append(latent_var)
        else:
            y_output = model(data)
        
        y_softmax = F.softmax(y_output, dim=1).cpu().detach().numpy()  # Convert to probabilities
        y_out.extend(y_softmax)

        pred = y_output.argmax(dim=1).cpu().numpy()

        y = data.y.cpu().data.numpy()
        y_pred.extend(pred)
        y_true.extend(y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    if predicts:
        if latent:
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


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    classify_test(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
