import torch
import torch.nn as nn
from torch.nn import functional as F
from feature_extract import DP_raw_fearure
from model.attention import Coattention, CrossAttention
from model.Embedding import Embeddings
from utils.tools import parser
import warnings
warnings.filterwarnings("ignore")
args = parser()

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(input_dim, 256, bias=True)
        self.linear2 = nn.Linear(256, output_dim, bias=False)
        self.dropout = nn.Dropout(args.dropout_rate)
        self.act = nn.ReLU()

    def forward(self, v):
        v = self.act(self.dropout(self.linear1(v)))
        v = self.act(self.linear2(v))
        return v

class MLPDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, binary=2):
        super(MLPDecoder, self).__init__()
        self.feat_drop = nn.Dropout(args.dropout_rate)
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, binary)

    def forward(self, x):
        x = F.relu(self.bn1(self.feat_drop(self.fc1(x))))
        x = F.relu(self.bn2(self.feat_drop(self.fc2(x))))
        x = self.fc3(x)
        return x


class SPACPI(nn.Module):
    def __init__(self,
                 N_fingerprints,
                 args=None):
        super(SPACPI, self).__init__()
        self.args = args

        if args.dropout_rate > 0:
            self.feat_drop = nn.Dropout(args.dropout_rate)
        else:
            self.feat_drop = lambda x: x

        self.dim = args.dim

        self.fp_embedding = Embeddings(608, self.dim, 180, args.dropout_rate)
        self.fc_D = MLP(input_dim=self.dim*2,output_dim=self.dim)
        self.DP_raw_fearure = DP_raw_fearure(N_fingerprints,dim=self.dim)

        self.co_attention = Coattention(dim=self.dim,k=int(self.dim/2))
        self.CrossAttention = CrossAttention(in_dim1=self.dim,in_dim2=self.dim,num_heads=2)

        self.mlp_classifier = MLPDecoder(in_dim=self.dim*2,hidden_dim=self.dim,out_dim=64)
        self.scale = torch.FloatTensor([10]).to(args.device)

    def forward(self, inputs,emb_init=None,pertubation_attention=None):
        p,compound_set,fp,y_true = inputs
        D0, P0 = self.DP_raw_fearure(compound_set, p, emb_init=None)

        '''Add disturbance to the original compound embedding'''
        if emb_init is not None:
            D0 = emb_init

        '''Using CrossAttention to attach fingerprint information to molecular graph information'''
        if args.use_fp:
            fp = self.fp_embedding(fp)
            D0_fp = self.CrossAttention(D0,fp)
            D0 = self.fc_D(torch.cat([D0,D0_fp],dim=-1))

        '''Calculate dot-product attention'''
        D, P, weights, attn_loss = self.co_attention(D0, P0, pertubation_attention)

        out = self.mlp_classifier(torch.cat([D,P], dim=1))

        '''Cross-Entropy Loss'''
        pred_loss = F.nll_loss(torch.log(F.softmax(out, dim=-1)), y_true)

        total_loss = pred_loss
        if pertubation_attention is not None:
            total_loss = total_loss + attn_loss*self.scale

        return out, total_loss




