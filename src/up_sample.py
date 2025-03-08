import numpy as np
import os
from collections import Counter 
import argparse
import time
import scanpy as sc

pathjoin = os.path.join


def get_parser(parser=None):
    
    if parser is None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Muraro.npz')  # Input file (NPZ format)
    parser.add_argument('-outdir', type=str, default='result/datasets/ ')  # Output directory for saving processed data
    parser.add_argument('-nsp', '--n_splits', type=int, default=5)  # Number of splits for cross-validation
    return parser


def get_train_index(args):
    """
    Loads the training data and creates an imputed training index with up-sampling.

    This function loads the 'seq_dict.npz' file which contains the indices of training samples.
    It performs up-sampling on the training data to balance class distribution and saves the imputed 
    indices to a file for future use.

    Args:
        args (Namespace): Command line arguments containing the dataset path, output directory, 
                          and number of splits for cross-validation.
    """
    expr_npz = args.expr
    save_folder = args.outdir
    n_splits = args.n_splits

    # Process the file name (e.g., 'Muraro') from the input NPZ file path
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]

    # Read the 'seq_dict' file which contains information on the dataset and splits
    seq_folder = pathjoin(save_folder, base_filename)
    seq_dict_file = pathjoin(seq_folder, 'seq_dict.npz')    
    seq_dict = np.load(seq_dict_file, allow_pickle=True) 
    label = seq_dict['label']

    # Create a list to store imputed training indices for each fold
    train_index_imputed = []

    # Perform up-sampling on the training data for each fold
    for k in range(n_splits):
        train_index = seq_dict[f'train_index_{k + 1}']
        label_train = label[train_index]
        
        # Apply up-sampling to the training labels
        impute_indexs = up_sample(label_train)
        train_index_imputed.append(impute_indexs)
    
    print("train_index_imputed: ")
    print(train_index_imputed)
    train_index_imputed_array = np.array(train_index_imputed, dtype=object)

    # Save the imputed training indices to a file
    train_index_imputed_file = pathjoin(seq_folder, f'{base_filename}_train_index_imputed.npy')
    np.save(train_index_imputed_file, train_index_imputed_array) 
    print("train_index_imputed_file has been saved!")


def up_sample(label, dup_odds=50, random_seed=2240):
    """
    Performs up-sampling on the provided labels to balance the class distribution.

    Args:
        label (array-like): Array of class labels.
        dup_odds (int, optional): Odds ratio to determine how much to up-sample the minority class. Default is 50.
        random_seed (int, optional): Random seed for reproducibility. Default is 2240.

    Returns:
        list: List of indices representing the up-sampled labels.
    """
    counter = Counter(label)
    max_num_types = max(counter.values())  # Get the maximum number of samples in any class
    num_sam = len(label)  # Total number of samples
    impute_indexs = np.arange(num_sam).tolist()  # Initialize the list of indices to include all samples
    np.random.seed(random_seed)  # Set the random seed for reproducibility
    
    # For each class, up-sample the minority classes to match the maximum class size
    for lab in np.unique(label):
        if max_num_types / np.sum(label == lab) > dup_odds:  # Check if the class is underrepresented
            impute_size = int(max_num_types / dup_odds) - np.sum(label == lab)  # Calculate how many samples to add
            print(f'Duplicate #celltype {lab} with {impute_size} cells')
            # Randomly select indices from the underrepresented class and add them to the training set
            impute_idx = np.random.choice(np.where(label == lab)[0], size=impute_size, replace=True).tolist()
            impute_indexs += impute_idx        
    
    return impute_indexs


if __name__ == '__main__':
    start_time = time.time()  
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    get_train_index(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} seconds")
