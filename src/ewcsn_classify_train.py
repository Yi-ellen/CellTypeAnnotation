import numpy as np
import os
import torch
import argparse
import time
from torch.optim.lr_scheduler import ExponentialLR
import torch.nn.functional as F
from sklearn.metrics import f1_score, accuracy_score
from torch_geometric.loader import DataLoader
from scipy.sparse import load_npz
import copy
from model import CSGNet
from datasets_ewcsn import MyDataset2

pathjoin = os.path.join


# Function to parse command line arguments
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-net', '--net', type=str, default='HumanNet-GSP')   # Gene interaction network (gold standard)
    parser.add_argument('-outdir', type=str, default='result/ewcsn_models')  # Output directory
    parser.add_argument('-cuda', '--cuda', type=bool, default=True)  # Use CUDA (GPU) for computation
    parser.add_argument('-bs', '--batch_size', type=int, default=32)  # Batch size for training
    parser.add_argument('-epoch', type=int, default=30)  # Number of epochs
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000)  # Number of high variance genes
    parser.add_argument('-ca', '--csn_alpha', type=float, default=0.01)   # Significance level for WCSN construction
    parser.add_argument('-addname', type=str, default="") 
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    parser.add_argument('-clist', '--channel_list', nargs='+', type=int, default=[256, 64], help='Model parameter list')  
    parser.add_argument('-mc', '--mid_channel', type=int, default=16)  # Mid channel dimension
    parser.add_argument('-gcdim1', '--global_conv1_dim', type=int, default=12)  # First global convolution dimension
    parser.add_argument('-gcdim2', '--global_conv2_dim', type=int, default=4)  # Second global convolution dimension
    
    return parser


# Function to train the classification model
def classify_train(args):
    """
    Train a classification model using scRNA-seq data and EWCSN.

    Args:
        args (Namespace): Command line arguments containing paths, hyperparameters, and settings.
    """
    expr_npz = args.expr
    net_filename = args.net
    models_folder = args.outdir
    csn_alpha = args.csn_alpha

    HVGs_num = args.high_var_genes
    addname = args.addname 
    n_splits = args.n_splits

    cuda_flag = args.cuda
    batch_size = args.batch_size
    mid_channel = args.mid_channel
    global_conv1_dim = args.global_conv1_dim
    global_conv2_dim = args.global_conv2_dim

    # Process file names, e.g., Baron_Human
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    device = torch.device('cuda' if torch.cuda.is_available() and cuda_flag else 'cpu')

    # Read sequence dictionary for cross-validation
    seq_folder = pathjoin('result/datasets/', base_filename)

    csn_data_folder = pathjoin(seq_folder, f"wcsn_a{csn_alpha}_hvgs{HVGs_num}")
    net_adj_file = pathjoin(seq_folder, f'{base_filename}_{net_filename}_net_adj.npz')    
    net_adj = load_npz(net_adj_file)

    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']
    str_labels = seq_dict['str_labels']

    # Get the number of classes
    class_num = len(np.unique(str_labels))

    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    # Get gene counts for each fold
    all_filtered_genes_array = np.load(all_filtered_genes_file, allow_pickle=True)
    genes_num_all = [len(row) for row in all_filtered_genes_array]

    train_index_imputed_file = pathjoin(seq_folder, f'{base_filename}_train_index_imputed.npy')
    train_index_imputed = np.load(train_index_imputed_file, allow_pickle=True)

    init_lr = 0.01
    max_epoch = args.epoch
    weight_decay = 1e-4  
    dropout_ratio = 0.1

    # Use weighted cross-entropy loss
    print('Use weighted cross-entropy...')
    label_type = np.unique(label.reshape(-1))
    alpha = np.array([np.sum(label == x) for x in label_type])
    alpha = np.max(alpha) / alpha
    alpha = np.clip(alpha, 1, 50)
    alpha = alpha / np.sum(alpha)
    loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor(alpha).float())
    loss_fn = loss_fn.to(device)

    # Train the model for each split
    for k in range(n_splits):
        k_fold = k + 1
        print("k_fold:", k_fold)

        cell_train_folder = os.path.join(csn_data_folder, f"train_f{k_fold}")
        filtered_genes_index = all_filtered_genes_array[k].astype(int)
        
        # Get submatrix of network adjacency
        cur_net_adj = net_adj.tocsr()[filtered_genes_index, :][:, filtered_genes_index].tocoo()  
        
        # Create train dataset and dataloader, get EWCSN
        train_dataset = MyDataset2(root=cell_train_folder, my_indices=train_index_imputed[k], net_adj=cur_net_adj)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=2)  

        gene_num = genes_num_all[k]
        channel_list = copy.deepcopy(args.channel_list)
        # Initialize and print the model
        model = CSGNet(in_channel=1, mid_channel=mid_channel, num_nodes=gene_num, out_channel=class_num, dropout_ratio=dropout_ratio, 
                       channel_list=channel_list, global_conv1_dim=global_conv1_dim, global_conv2_dim=global_conv2_dim).to(device)
        print(model)

        optimizer = torch.optim.Adam(model.parameters(), lr=init_lr, weight_decay=weight_decay)
        scheduler = ExponentialLR(optimizer, gamma=0.8)

        # Train and test the model
        for epoch in range(1, max_epoch):
            train_loss, train_acc, train_f1 = train(model, optimizer, train_loader, device, loss_fn)
            scheduler.step()
            lr = optimizer.param_groups[0]['lr']
            print('epoch\t%03d, lr: %.06f, loss: %.06f, Train-acc: %.04f, Train-f1: %.04f' % (epoch, lr, train_loss, train_acc, train_f1))
            print(f'Epoch {epoch + 1}, Learning Rate: {scheduler.get_last_lr()[0]}')

        # Save the model
        model_file = pathjoin(models_folder, f'{base_filename}_{net_filename}_a{csn_alpha}_hvgs{HVGs_num}_model{addname}{k_fold}.pth')
        torch.save(model, model_file)



def train(model, optimizer, train_loader, device, loss_fn=None, verbose=False):
    model.train()
    loss_all = 0
    iters = len(train_loader)
    y_pred = []
    y_true = []

    for idx, data in enumerate(train_loader):
        data = data.to(device)
        if verbose:
            print(data.x.shape, data.y.shape, data.edge_index.shape, data.edge_weight.shape)
        optimizer.zero_grad()
        output = model(data)
        if loss_fn is None:
            loss = F.cross_entropy(output, data.y.reshape(-1), weight=None)
        else:
            loss = loss_fn(output, data.y.reshape(-1))
        loss.backward()
        optimizer.step()

        loss_all += loss.item()

        with torch.no_grad():  # Disable gradient calculation for inference
            pred = output.argmax(dim=1).cpu().numpy()  # Predicted labels
            y = data.y.cpu().numpy()  # True labels

        y_pred.extend(pred)
        y_true.extend(y)

    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    return loss_all / iters, acc, f1


# Main execution block
if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    classify_train(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")

