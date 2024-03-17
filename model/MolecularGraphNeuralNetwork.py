import torch
import torch.nn as nn
import warnings
from model.Embedding import Embeddings
from model.LayerNorm import LayerNorm
warnings.filterwarnings("ignore")
from utils.tools import parser
args = parser()


class MolecularGraphNeuralNetwork(nn.Module):
    def __init__(self, N_fingerprints, dim, layer_hidden, layer_output):
        super(MolecularGraphNeuralNetwork, self).__init__()
        self.embed_fingerprint = nn.Embedding(N_fingerprints, dim)
        self.max_d = args.Dmax
        self.demb = Embeddings(N_fingerprints, dim, self.max_d, args.dropout_rate)

        self.layer_hidden = layer_hidden
        self.layer_output = layer_output
        self.W_gnn1 = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_gnn2 = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.W_gnn3 = nn.ModuleList([nn.Linear(dim, dim)
                                     for _ in range(layer_hidden)])
        self.do = nn.Dropout(args.dropout_rate)
        self.LayerNorm = LayerNorm(dim)

    def gat(self, inputs):
        fingerprints, adjacencies, molecular_sizes = inputs

        fingerprints = torch.stack(fingerprints)
        adjacencies = torch.stack(adjacencies)

        fingerprint_vectors = self.demb(fingerprints)  # [1,max_len,dim]

        xs = fingerprint_vectors

        for l in range(self.layer_hidden):
            hs1 = torch.relu(self.W_gnn1[l](xs))  # [bs,N_fingerprints,dim]
            hs2 = torch.relu(self.W_gnn2[l](xs))  # [bs,N_fingerprints,dim]
            weights = torch.matmul(hs1, hs2.transpose(1, 2))

            attn = weights.mul(adjacencies)  # [bs,N_fingerprints,N_fingerprints]

            hs3 = torch.relu(self.W_gnn3[l](xs))

            xs = xs + self.LayerNorm(torch.matmul(attn, hs3))
            xs = self.do(xs)

        molecular_vectors = xs

        return molecular_vectors

    def forward(self, data_batch):
        inputs = data_batch
        molecular_vectors = self.gat(inputs)
        return molecular_vectors