import torch
import torch.nn as nn
from model.MolecularGraphNeuralNetwork import MolecularGraphNeuralNetwork
from model.ProteinCNN import ProteinCNN
import warnings
warnings.filterwarnings("ignore")
from utils.tools import parser
args = parser()

class DP_raw_fearure(nn.Module):
    def __init__(self,N_fingerprints,dim):
        super(DP_raw_fearure, self).__init__()
        protein_emb_dim = dim
        num_filters = [dim,dim,dim]
        kernel_size = [5,5,5]
        protein_padding = True
        self.drug_extractor = MolecularGraphNeuralNetwork(N_fingerprints, dim=dim, layer_hidden=3, layer_output=3)
        self.protein_extractor = ProteinCNN(protein_emb_dim, num_filters, kernel_size, protein_padding)
        self.device = args.device

    def forward(self,v_d, v_p, emb_init=None):
        v_d = list(zip(*v_d[:]))
        trans_d = self.drug_extractor(v_d, emb_init=emb_init)
        trans_p = self.protein_extractor(v_p.to(self.device))
        return trans_d, trans_p  # torch.



