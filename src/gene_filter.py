import numpy as np
import pandas as pd
import os
import argparse
import time
import scanpy as sc

pathjoin = os.path.join

def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    # Input file containing the gene expression dataset
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz',
                        help='Path to the input gene expression .npz file')
    # Output directory to save filtered gene results
    parser.add_argument('-outdir', type=str, default='result/datasets/',
                        help='Directory to save the filtered gene results')
    # Number of highly variable genes to filter
    parser.add_argument('-hvgs', '--high_var_genes', type=int, default=2000,
                        help='Number of highly variable genes to retain')
    return parser


def gene_filter(args):
    """
    Filter highly variable genes (HVGs) from the dataset and save the results.

    Args:
        args (argparse.Namespace): Parsed command-line arguments containing:
            - expr (str): Path to the input .npz file.
            - outdir (str): Path to the output directory.
            - high_var_genes (int): Number of highly variable genes to retain.
    """
    expr_npz = args.expr
    save_folder = args.outdir
    HVGs_num = args.high_var_genes

    # Extract the base filename from the input file path (e.g., 'Muraro.npz' -> 'Muraro')
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    # Define the output directory for saving filtered gene results
    seq_folder = pathjoin(save_folder, base_filename)

    # Process the gene expression data to compute log-transformed expression values
    logExpr1 = get_logExpr2(expr_npz)  # logExpr: row-cell, column-gene  

    # Filter highly variable genes based on log-transformed expression data
    filtered_HVGs_index = filtered_HVGs(logExpr1, HVGs_num)

    # Store filtered genes for each fold
    all_filtered_genes = []  
    
    for k in range(args.n_splits):
        k_fold = k + 1
        print("k_fold: ", k_fold) 
        all_filtered_genes.append(filtered_HVGs_index)

    # Convert all filtered genes into a 2D array
    all_filtered_genes_array = np.array(all_filtered_genes, dtype=object)

    # Save the filtered genes to a .npy file
    all_filtered_genes_file = pathjoin(seq_folder, f'{base_filename}_filtered_hvgs{HVGs_num}.npy')
    np.save(all_filtered_genes_file, all_filtered_genes_array)    

    print('The filtered_genes is generated')   


def get_logExpr2(expr_npz):
    """
    Compute log-transformed expression values for the input dataset.

    Args:
        expr_npz (str): Path to the input .npz file containing raw gene expression data.

    Returns:
        np.array: Log-transformed gene expression matrix with rows as cells and columns as genes.
    """
    data = np.load(expr_npz, allow_pickle=True)
    # Extract raw gene expression data (row-cell, column-gene)
    countExpr = data['count']  
    print("raw (cells, genes): ", countExpr.shape)

    # Normalize each row so that the sum equals 1e6
    row_sums = countExpr.sum(axis=1, keepdims=True)
    normalized_data = 1e6 * countExpr / row_sums

    logExpr = np.log1p(normalized_data) 

    return logExpr


def filtered_HVGs(logExpr, HVGs_num):
    """
    Filter highly variable genes (HVGs) from the log-transformed expression data.

    Args:
        logExpr (np.array): Log-transformed gene expression matrix.
        HVGs_num (int): Number of highly variable genes to retain.

    Returns:
        np.array: Indices of highly variable genes.
    """
    if HVGs_num == 0:
        print("high variable genes num: 0")
        return np.array([])

    # Create an AnnData object from the log-transformed expression matrix
    adata = sc.AnnData(X=logExpr)

    # Filter highly variable genes and retain the top `HVGs_num` genes
    sc.pp.highly_variable_genes(adata, n_top_genes=HVGs_num)

    # Retrieve the indices of highly variable genes
    hvgs = adata.var['highly_variable']   
    hvgs_indices = np.where(hvgs)[0]    

    return hvgs_indices


if __name__ == '__main__':
    """
    Main script execution: Parse arguments, filter highly variable genes, and save the results.
    """
    start_time = time.time()  

    # Parse command-line arguments
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    # Perform highly variable gene filtering
    gene_filter(args)

    # Print the total runtime of the script
    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
