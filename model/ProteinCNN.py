import torch
import torch.nn as nn
from Embedding import Embeddings
import warnings
warnings.filterwarnings("ignore")
from utils.tools import parser
args = parser()

class ProteinCNN(nn.Module):
    def __init__(self, embedding_dim, num_filters, kernel_size, padding=True):
        super(ProteinCNN, self).__init__()
        self.pemb = Embeddings(16693, embedding_dim, 545, args.dropout_rate)
        if padding:
            self.embedding = nn.Embedding(16693, embedding_dim, padding_idx=0)
        else:
            self.embedding = nn.Embedding(16693, embedding_dim)
        in_ch = [embedding_dim] + num_filters
        # in_ch = num_filters
        self.in_ch = in_ch[-1]
        kernels = kernel_size

        self.convs = nn.ModuleList(
                        [nn.Conv1d(in_channels=in_ch[i], out_channels=in_ch[i+1], kernel_size=kernels[i], padding=(kernels[i] - 1) // 2) for i in
                         range(len(kernels))])  # convolutional layers
        self.dropout = nn.Dropout(args.dropout_rate)
        self.scale = torch.FloatTensor([0.5]).to(args.device)

    def forward(self, v):
        v = self.pemb(v.long())
        v = v.transpose(2, 1)
        conved = v
        for i, conv in enumerate(self.convs):
            conved = conv(conved)
            conved = self.dropout(conved)

        v = (v + conved) * self.scale
        v = v.permute(0, 2, 1)


        return v#.squeeze(1)
