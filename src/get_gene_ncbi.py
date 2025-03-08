import numpy as np
import pandas as pd
import argparse
import time
import os

pathjoin = os.path.join


def get_parser(parser=None):
    # Set up argument parser for input arguments
    if parser == None:
        parser = argparse.ArgumentParser()
    parser.add_argument('-expr', type=str, default='data/pre_data/scRNAseq_datasets/Baron_Human.npz')  # Input file, npz file
    parser.add_argument('-outdir', type=str, default='result/datasets/')  # Output directory
    parser.add_argument('-gn', '--gene_name', type=str, default='hgnc')  # Gene name format: 'hgnc', 'mgi', 'ensembl', or 'ncbi'
    parser.add_argument('-species', type=str, default='human')  # Species: 'human' or 'mouse'
    return parser


def get_gene_ncbi(args):
    # Load and process gene symbols from the input dataset
    expr_npz = args.expr
    gene_name = args.gene_name
    species = args.species
    save_folder = args.outdir

    # Extract the base filename (without extension)
    base_filename = os.path.basename(expr_npz)
    base_filename = os.path.splitext(base_filename)[0]   

    data = np.load(expr_npz, allow_pickle=True)  # Load data from npz file
    genes = data['gene_symbol']  # Extract gene symbols
    print("gene num:")
    print(len(genes))  # Print number of genes
    genes_new = deal_gene(genes, base_filename)  # Deal with dataset-specific gene symbols

    # Map gene symbols to Entrez Gene IDs based on the specified format
    if gene_name == 'hgnc' and species == 'human':
        genes_ncbi = get_gene_ncbi_from_hgnc(genes_new)
    elif gene_name == 'mgi' and species == 'mouse':
        genes_ncbi = get_gene_ncbi_from_mgi(genes_new)
    elif gene_name == 'ensembl' and species == 'human':
        genes_ncbi = get_gene_ncbi_from_ensembl(genes_new)
    elif gene_name == 'ncbi':
        genes_ncbi = genes  # If gene name is 'ncbi', no conversion is needed

    # Save the results to the specified output directory
    os.makedirs(pathjoin(save_folder, base_filename), exist_ok=True)
    seq_folder = pathjoin(save_folder, base_filename)
    np.save(os.path.join(seq_folder, 'genes_ncbi.npy'), genes_ncbi)  # Save gene list as .npy file


def deal_gene(genes, base_filename):
    # Handle dataset-specific gene name adjustments
    if base_filename == 'Baron_Human':
        genes = [gene.replace('.', '-') for gene in genes]
    elif base_filename == 'Muraro':
        genes = [gene.split('__')[0] for gene in genes]
    elif base_filename == 'Segerstolpe':
        pass
    elif base_filename == 'Xin':
        genes = [gene.replace('.', '-') for gene in genes]
    elif base_filename == 'Zhang_T':
        pass
    elif base_filename == 'Kang_ctrl':
        pass
    elif base_filename == 'Kang_stim':
        pass
    elif base_filename == 'Zheng68K':
        genes = [gene.rsplit('.', 1)[0] for gene in genes]
    elif base_filename == 'Baron_Mouse':
        genes = [gene.replace('.', '-') for gene in genes]
    elif base_filename == 'TM':
        pass
    elif base_filename == 'AMB':
        pass
    return genes


def get_gene_ncbi_from_hgnc(genes):
    # Map gene symbols to Entrez Gene IDs using HGNC data
    hgnc_path = 'data/HGNC/hgnc/hgnc_complete_set.txt'
    hgnc_df = pd.read_csv(hgnc_path, sep='\t', usecols=[0, 1, 10, 18], dtype={18: str})

    """
    columns:
    Index(['hgnc_id', 'symbol', 'name', 'locus_group', 'locus_type', 'status',
        'location', 'location_sortable', 'alias_symbol', 'alias_name',
        'prev_symbol', 'prev_name', 'gene_group', 'gene_group_id',
        'date_approved_reserved', 'date_symbol_changed', 'date_name_changed',
        'date_modified', 'entrez_id', 'ensembl_gene_id', 'vega_id', 'ucsc_id',
        'ena', 'refseq_accession', 'ccds_id', 'uniprot_ids', 'pubmed_id',
        'mgd_id', 'rgd_id', 'lsdb', 'cosmic', 'omim_id', 'mirbase', 'homeodb',
        'snornabase', 'bioparadigms_slc', 'orphanet', 'pseudogene.org',
        'horde_id', 'merops', 'imgt', 'iuphar', 'kznf_gene_catalog',
        'mamit-trnadb', 'cd', 'lncrnadb', 'enzyme_id',
        'intermediate_filament_db', 'rna_central_ids', 'lncipedia', 'gtrnadb',
        'agr', 'mane_select', 'gencc'],
        dtype='object')

    """


    # Load withdrawn gene symbols
    withdrawn_path = 'data/HGNC/hgnc/withdrawn.txt'
    withdrawn_df = pd.read_csv(withdrawn_path, sep='\t')

    sym2ncbi = []  # List to store Entrez Gene IDs
    not_exist = []  # List to store genes not found in HGNC

    # Process each gene symbol and map it to an Entrez Gene ID
    for idx, gene in enumerate(genes):
        print("Current gene: ", gene)
        if gene in hgnc_df["symbol"].values:
            sym2ncbi.append(hgnc_df[hgnc_df["symbol"] == gene]["entrez_id"].iloc[0])
            print("cur_sym", sym2ncbi[idx])
        elif gene in withdrawn_df["WITHDRAWN_SYMBOL"].values:
            if(withdrawn_df[withdrawn_df["WITHDRAWN_SYMBOL"] == gene]["STATUS"].iloc[0] == "Entry Withdrawn"):
                sym2ncbi.append(None)
                print("withdrawn_sym", sym2ncbi[idx])
            else:
                mapped_sym = withdrawn_df[withdrawn_df["WITHDRAWN_SYMBOL"] == gene].iloc[0, 3].split('|')[1]
                sym2ncbi.append(hgnc_df[hgnc_df["symbol"] == mapped_sym]["entrez_id"].iloc[0])
                print("merged_sym", sym2ncbi[idx])
        else:
            count = 0
            for index, prev_symbols in hgnc_df["prev_symbol"].items():
                if pd.isna(prev_symbols):
                    continue
                symbols = prev_symbols.split('|')
                if gene in symbols:
                    count += 1
                    sym2ncbi.append(hgnc_df.loc[index, "entrez_id"])
                    print("pre_sym", sym2ncbi[idx])
                    break
            if(count == 0):
                not_exist.append(gene)
                sym2ncbi.append(None)
                print("not_exist")

    print("len(genes): ", len(genes))
    print("len(sym2ncbi): ", len(sym2ncbi))
    print("len(not_exist): ", len(not_exist))

    return sym2ncbi


def get_gene_ncbi_from_mgi(genes):
    # Map gene symbols to Entrez Gene IDs using MGI data
    MRK_all_path = 'data/MGI/MRK_List1.rpt'
    MRK_all_df = pd.read_csv(MRK_all_path, usecols=[6, 7, 12, 13], sep='\t', dtype=str)
    MRK_all_df = MRK_all_df.drop_duplicates()

    # Separate data by status: 'O' for active, 'W' for withdrawn
    MRK_O_df = MRK_all_df[MRK_all_df['Status'] == 'O']
    MRK_W_df = MRK_all_df[MRK_all_df['Status'] == 'W']

    # Load Entrez Gene IDs from MGI
    file_path = 'data/MGI/MGI_EntrezGene.rpt'
    mgi_ncbi_df = pd.read_csv(file_path, sep='\t', usecols=[0, 1, 2, 8, 9], header=None, index_col=False, dtype=str)
    mgi_ncbi_df.columns = ['MGI Marker Accession ID', 'Marker Symbol', 'Status', 'Entrez Gene ID', 'Synonyms']

    """
    columns:
    0 MGI Marker Accession ID	
    1 Marker Symbol	
    2 Status	
    3 Marker Name	
    4 cM Position	
    5 Chromosome	
    6 Type Gene DNA Segment Other Genome Feature Complex/Cluster/RegionmicroRNA	
    7 Secondary Accession IDs (|-delimited)	
    8 Entrez Gene ID	
    9 Synonyms(|-delimited)	
    10 Feature Types (|-delimited)	
    11 Genome Coordinate Start	
    12 Genome Coordinate End	
    13 Strand	
    14 BioTypes(|-delimited)
    """

    sym2ncbi = []  # List to store Entrez Gene IDs
    not_exist = []  # List to store genes not found in MGI

    # Process each gene symbol and map it to an Entrez Gene ID
    for gene in genes:
        print("Current gene: ", gene)
        if gene in MRK_O_df['Marker Symbol'].values:
            if gene in mgi_ncbi_df['Marker Symbol'].values:
                ncbi = mgi_ncbi_df[mgi_ncbi_df['Marker Symbol'] == gene]['Entrez Gene ID'].iloc[0]
                sym2ncbi.append(ncbi)
                print("cur_sym: ", ncbi)
            else:
                sym2ncbi.append(None)
                print("cur_sym: ", None)
        elif gene in MRK_W_df['Marker Symbol'].values:
            count = 0
            mapped_syms = MRK_W_df[MRK_W_df['Marker Symbol'] == gene]['Current Marker Symbol (if withdrawn)']
            for mapped_sym in mapped_syms:
                if pd.notna(mapped_sym):
                    count += 1
                    print('mapped_sym:', mapped_sym)
                    if mapped_sym in mgi_ncbi_df['Marker Symbol'].values:
                        ncbi = mgi_ncbi_df[mgi_ncbi_df['Marker Symbol'] == mapped_sym]['Entrez Gene ID'].iloc[0]
                        sym2ncbi.append(ncbi)
                        print('merged_sym: ', ncbi)
                        break
            if count == 0:
                sym2ncbi.append(None)
                print('withdrawn!')
            else:
                pass
        else:
            not_exist.append(gene)
            sym2ncbi.append(None)
            print('not exist!')

    return sym2ncbi


def get_gene_ncbi_from_ensembl(genes):
    # Map Ensembl gene IDs to Entrez Gene IDs
    hgnc_path = 'data/HGNC/hgnc/hgnc_complete_set.txt'
    hgnc_df = pd.read_csv(hgnc_path, sep='\t', usecols=[0, 1, 10, 18, 19], dtype=str)

    # Remove rows without Ensembl gene ID
    hgnc_df = hgnc_df.dropna(subset=['ensembl_gene_id'])

    ens2ncbi = []  # List to store Entrez Gene IDs
    not_exist = []  # List to store genes not found
    exist = []  # List to store genes with valid Entrez IDs

    # Process each Ensembl gene symbol
    for gene in genes:
        print("Current ensembl gene: ", gene)
        if gene in hgnc_df["ensembl_gene_id"].values:
            ncbi = hgnc_df[hgnc_df["ensembl_gene_id"] == gene]["entrez_id"].iloc[0]
            if pd.isna(ncbi):
                not_exist.append(gene)
                ens2ncbi.append(None)
                print("not_exist")
            else:
                ens2ncbi.append(ncbi)
                exist.append(gene)
                print("entrez_id", ncbi)
        else:
            not_exist.append(gene)
            ens2ncbi.append(None)
            print("not_exist")

    return ens2ncbi


if __name__ == '__main__':
    start_time = time.time()
    parser = get_parser()
    args = parser.parse_args()
    print('args:', args)

    genes_ncbi = get_gene_ncbi(args)

    end_time = time.time()
    print(f"Code run time: {end_time - start_time} s")
