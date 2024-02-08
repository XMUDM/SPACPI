from collections import defaultdict
import numpy as np
from rdkit import Chem
import torch.nn.functional as F
import torch
from utils.tools import parser,shuffle_dataset,split_dataset
import pandas as pd
import torch
from subword_nmt.apply_bpe import BPE
import codecs
args = parser()


def load_dataset(dir):
    df = pd.read_csv(dir,
                    header=None,
                    delimiter=' ',
                    names=['smiles', 'sequence', 'label'])
    drug_smiles = df['smiles']
    protein_seqs = df['sequence']
    cpi_labels = df['label']
    compouns_set, N_fingerprints = create_compound_datasets(drug_smiles, radius=1)
    return drug_smiles, protein_seqs, cpi_labels, compouns_set, N_fingerprints

def dataset_segmentation(dataset,cpi_dataset_name):
    if cpi_dataset_name == 'GPCR':
        dataset_ = dataset[:13022]
        dataset_test = dataset[13022:]
        dataset_ = shuffle_dataset(dataset_, 1234)
        dataset_dev, dataset_train = split_dataset(dataset_, 0.1)

    elif cpi_dataset_name == 'kinase':
        dataset_ = dataset[:91465]
        dataset_test = dataset[91465:]
        dataset_ = shuffle_dataset(dataset_, 1234)
        dataset_dev, dataset_train = split_dataset(dataset_, 0.1)

    else:
        dataset = shuffle_dataset(dataset, 1234)
        dataset_test, dataset_ = split_dataset(dataset, 0.1)
        dataset_dev, dataset_train = split_dataset(dataset_, 0.1)

    return dataset_train,dataset_dev,dataset_test


def create_atoms(mol, atom_dict):
    """Transform the atom types in a molecule (e.g., H, C, and O)
    into the indices (e.g., H=0, C=1, and O=2).
    Note that each atom index considers the aromaticity.
    """
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    # print(atoms)
    for a in mol.GetAromaticAtoms():
        i = a.GetIdx()
        atoms[i] = (atoms[i], 'aromatic')
    atoms = [atom_dict[a] for a in atoms]
    return np.array(atoms)


def create_ijbonddict(mol, bond_dict):
    """Create a dictionary, in which each key is a node ID
    and each value is the tuples of its neighboring node
    and chemical bond (e.g., single and double) IDs.
    """
    i_jbond_dict = defaultdict(lambda: [])
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bond = bond_dict[str(b.GetBondType())]
        i_jbond_dict[i].append((j, bond))
        i_jbond_dict[j].append((i, bond))
    return i_jbond_dict


def extract_fingerprints(radius, atoms, i_jbond_dict,
                         fingerprint_dict, edge_dict):
    """Extract the fingerprints from a molecular graph
    based on Weisfeiler-Lehman algorithm.
    """

    if (len(atoms) == 1) or (radius == 0):
        nodes = [fingerprint_dict[a] for a in atoms]

    else:
        nodes = atoms
        i_jedge_dict = i_jbond_dict

        for _ in range(radius):

            """Update each node ID considering its neighboring nodes and edges.
            The updated node IDs are the fingerprint IDs.
            """
            nodes_ = []
            for i, j_edge in i_jedge_dict.items():
                neighbors = [(nodes[j], edge) for j, edge in j_edge]
                fingerprint = (nodes[i], tuple(sorted(neighbors)))
                nodes_.append(fingerprint_dict[fingerprint])

            """Also update each edge ID considering
            its two nodes on both sides.
            """
            i_jedge_dict_ = defaultdict(lambda: [])
            for i, j_edge in i_jedge_dict.items():
                for j, edge in j_edge:
                    both_side = tuple(sorted((nodes[i], nodes[j])))
                    edge = edge_dict[(both_side, edge)]
                    i_jedge_dict_[i].append((j, edge))

            nodes = nodes_
            i_jedge_dict = i_jedge_dict_

    return np.array(nodes)

def create_compound_datasets(smiles, radius):
    """Initialize x_dict, in which each key is a symbol type
    (e.g., atom and chemical bond) and each value is its index.
    """
    atom_dict = defaultdict(lambda: len(atom_dict))
    bond_dict = defaultdict(lambda: len(bond_dict))
    fingerprint_dict = defaultdict(lambda: len(fingerprint_dict))
    edge_dict = defaultdict(lambda: len(edge_dict))
    dataset=[]

    for i in range(len(smiles)):

            smile = smiles[i]
            """Create each data with the above defined functions."""
            mol = Chem.AddHs(Chem.MolFromSmiles(smile))
            atoms = create_atoms(mol, atom_dict)
            molecular_size = len(atoms)
            i_jbond_dict = create_ijbonddict(mol, bond_dict)
            # print(i,smile,len(atoms),len(i_jbond_dict))
            fingerprints = extract_fingerprints(radius, atoms, i_jbond_dict,
                                                fingerprint_dict, edge_dict)
            adjacency = Chem.GetAdjacencyMatrix(mol)

            """Transform the above each data of numpy
            to pytorch tensor on a device (i.e., CPU or GPU).
            """
            max_length = args.Cmax
            fingerprints = torch.LongTensor(fingerprints).to(args.device)
            adjacency = torch.FloatTensor(adjacency).to(args.device)

            if fingerprints.shape[0]>=max_length:
                fingerprints = fingerprints[:max_length]
                adjacency = adjacency[:max_length,:max_length]
            else:
                fingerprints = F.pad(fingerprints, (0, max_length - len(fingerprints)), value=0)
                adjacency = F.pad(adjacency, (0, max_length - adjacency.size(1), 0, max_length - adjacency.size(0)),value=0)

            dataset.append((fingerprints, adjacency, molecular_size))

    N_fingerprints = len(fingerprint_dict)

    return dataset, N_fingerprints

def create_compound_datasets_test(compounds_set,idx):
    dataset = []
    for i in range(len(idx)):
        dataset.append(compounds_set[idx[i]])
    return dataset



vocab_path = './utils/protein_codes_uniprot.txt'
bpe_codes_protein = codecs.open(vocab_path)
pbpe = BPE(bpe_codes_protein, merges=-1, separator='')
sub_csv = pd.read_csv('./utils/subword_units_map_uniprot.csv')

idx2word_p = sub_csv['index'].values
words2idx_p = dict(zip(idx2word_p, range(0, len(idx2word_p))))

args = parser()
def protein2emb_encoder(x):
    max_p = args.Pmax
    t1 = pbpe.process_line(x).split()  # split
    try:
        i1 = np.asarray([words2idx_p[i] for i in t1])  # index
        # i1 = np.asarray([0 if 'X' in i else words2idx_p[i] for i in t1])
    except:
        i1 = np.array([0])
        # print(x)

    l = len(i1)

    if l < max_p:
        i = np.pad(i1, (0, max_p - l), 'constant', constant_values=0)
        input_mask = ([1] * l) + ([0] * (max_p - l))
    else:
        i = i1[:max_p]
        input_mask = [1] * max_p

    return i#, np.asarray(input_mask), t1
