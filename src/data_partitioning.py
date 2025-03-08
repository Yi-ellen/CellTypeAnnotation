import numpy as np
import pandas as pd
import os
from sklearn.model_selection import StratifiedKFold
from collections import Counter 
import argparse
import time

pathjoin = os.path.join

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz',
                        help='Path to the input gene expression .npz file')
    parser.add_argument('-outdir', type=str, default='result/datasets/',
                        help='Directory to save the output results')
    parser.add_argument('-nsp', '--n_splits', type=int, default=5,
                        help='Number of splits for cross-validation')
    return parser


def pre_processing(args):
    """
    Preprocess the input dataset and generate cross-validation splits.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - expr (str): Path to the input .npz file.
            - outdir (str): Path to the output directory.
            - n_splits (int): Number of cross-validation splits.
    """
    expr_npz = args.expr
    save_folder = args.outdir
    n_splits = args.n_splits   

    # Extract the base filename from the input file path (e.g., 'Muraro.npz' -> 'Muraro')
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    # Create the output directory if it doesn't already exist
    os.makedirs(pathjoin(save_folder, base_filename), exist_ok=True)
    seq_folder = pathjoin(save_folder, base_filename)

    # Define the file path for the cross-validation split results
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    

    if not os.path.exists(seq_dict_file):
        # Load barcodes, gene symbols, labels, and string labels from the .npz file
        barcodes, genes_symbol, label, str_labels = get_info2(expr_npz)
        
        # Print the number of unique cell types
        cell_type_num = len(str_labels)
        print("cell_type_num: ", cell_type_num)

        # Print the total number of cells in the dataset
        cell_num = len(barcodes)
        print("cell_num: ", cell_num)

        # Generate and save the cross-validation splits
        splits_and_save(genes_symbol, str_labels, barcodes, n_splits, label, seq_folder)
    
    print("splits_and_save is ok!")


def splits_and_save(genes_symbol, str_labels, barcodes, n_splits, label, seq_folder):
    """
    Generate cross-validation splits and save them in a dictionary.

    Args:
        genes_symbol (array): Array of gene symbols.
        str_labels (array): Array of string labels for cell types.
        barcodes (array): Array of unique cell identifiers.
        n_splits (int): Number of cross-validation splits.
        label (array): Array of numerical labels for cells.
        seq_folder (str): Directory where the cross-validation results will be saved.

    Returns:
        dict: A dictionary containing the cross-validation split indices, gene symbols, 
              labels, and barcodes.
    """
    seq_dict = {}
    # Add gene symbols, cell labels, and barcodes to the dictionary
    seq_dict['gene_symbol'] = genes_symbol
    seq_dict['label'] = label
    seq_dict['str_labels'] = str_labels
    seq_dict['barcode'] = barcodes

    # Initialize StratifiedKFold for generating cross-validation splits
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
    k_fold = 0 
    
    for tr, ts in kf.split(X=label, y=label):
        k_fold += 1
        print("k_fold: ", k_fold)
        train_index = tr 
        test_index = ts     
        label_train = label[train_index]

        # Print the number of training and testing samples for the current fold
        print("train_num: ", len(label_train))
        print("test_num: ", len(test_index))

        # Save training and testing indices for the current fold
        seq_dict[f'train_index_{k_fold}'] = train_index
        seq_dict[f'test_index_{k_fold}'] = test_index
    
    # Save the dictionary as a .npz file
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')
    np.savez(seq_dict_file, **seq_dict)
    return seq_dict                 


def get_info2(expr_file):
    """
    Load and extract information from the input .npz gene expression file.

    Args:
        expr_file (str): Path to the input .npz file.

    Returns:
        tuple: Contains the following:
            - barcodes (array): Unique identifiers for cells.
            - genes_symbol (array): Gene symbols.
            - label (array): Numerical labels for cells.
            - str_labels (array): String labels for cell types.
    """
    # Load the .npz file
    data = np.load(expr_file, allow_pickle=True)
    # Extract gene symbols, labels, string labels, and barcodes
    genes_symbol = data['gene_symbol']
    label = data['label']
    str_labels = data['str_labels']
    barcodes = data['barcode']

    return barcodes, genes_symbol, label, str_labels


if __name__ == '__main__':
    start_time = time.time()  

    # Parse command-line arguments
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    # Preprocess the dataset and generate cross-validation splits
    pre_processing(args)

    # Print the runtime of the script
    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
