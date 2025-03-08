import numpy as np
import pandas as pd
import os
from scipy.sparse import lil_matrix, save_npz
import argparse
import time
import scanpy as sc

pathjoin = os.path.join

# Function to parse command line arguments
def get_parser(parser=None):
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr',type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-net','--net',type=str, default='data/pre_data/network/HumanNet-GSP.tsv')   # Gene interaction network (gold standard)
    parser.add_argument('-outdir', type=str, default='result/dataset/')  # Output directory
    return parser


# Main function to process the network adjacency matrix
def net_adj(args):
    expr_npz = args.expr
    net_file = args.net    
    save_folder = args.outdir
    # Process the expression file name, e.g., Muraro
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    # Process the network file name, e.g., HumanNet-GSP
    net_filename = os.path.basename(net_file)
    net_filename = os.path.splitext(net_filename)[0]   

    seq_folder = pathjoin(save_folder, base_filename)
    
    genes_file = pathjoin(seq_folder, 'genes_ncbi.npy')
    genes_id = np.load(genes_file, allow_pickle=True)
    
    # Get the gene interaction network adjacency matrix
    net_adj = get_net_adj(net_file, genes_id)

    # Save the adjacency matrix to a file
    net_adj_file = pathjoin(seq_folder, f'{base_filename}_{net_filename}_net_adj.npz')    
    save_npz(net_adj_file, net_adj)


def get_net_adj(net_file, genes_id):
    """
    Generates the adjacency matrix for gene interactions.

    Args:
        net_file (str): Path to the gene interaction network file (e.g., TSV file).
        genes_id (array-like): List or array of gene identifiers used in the matrix.

    Returns:
        coo_matrix: A sparse adjacency matrix representing gene interactions.
    """
    net_df = pd.read_csv(net_file, header=None, index_col=None, sep='\t', dtype=str)
    net_df.columns = ['node1', 'node2']  # Rename columns for clarity
    # Generate the adjacency matrix from the gene interaction data
    net_adj = get_adjacency(net_df, genes_id)
    return net_adj



def get_adjacency(graph_edge_df, genes):
    """
    Creates an adjacency matrix from a DataFrame of gene interaction edges.

    Args:
        graph_edge_df (pd.DataFrame): A DataFrame containing gene interaction edges with columns 'node1' and 'node2'.
        genes (array-like): List or array of gene identifiers to map interactions to indices.

    Returns:
        coo_matrix: A sparse upper triangular adjacency matrix representing the gene interactions.
    """
    # Create an upper triangular adjacency matrix, excluding diagonal elements (no self-interactions)
    gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
    num_genes = len(genes)
    adj = lil_matrix((num_genes, num_genes), dtype=int)  # Initialize sparse matrix
    # Add interaction edges to the adjacency matrix
    count = 0
    for _, row in graph_edge_df.iterrows():
        gene1 = row['node1']
        gene2 = row['node2']

        if gene1 not in gene_to_index or gene2 not in gene_to_index:
            continue
        count += 1
        idx1 = gene_to_index[gene1]
        idx2 = gene_to_index[gene2]
        if(idx2 > idx1):
            adj[idx1, idx2] = 1
        elif(idx2 < idx1):
            adj[idx2, idx1] = 1

    adj = adj.tocoo()  # Convert matrix to COO format (sparse matrix)
    print('edge_nums:', len(adj.row))  # Print the number of edges in the network
    return adj



# Main execution block
if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()  # Get command line arguments
    args = parser.parse_args()
    print('args:', args)
    net_adj(args)  # Process the network adjacency matrix

    end_time = time.time()
    print(f"Code execution time: {end_time - start_time} seconds")  # Output the runtime
