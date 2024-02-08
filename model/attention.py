import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tools import stable_kl

'''Dot-Product Attention'''
class Coattention(nn.Module):
    def __init__(self, dim, k):
        super(Coattention, self).__init__()

        self.linear_a = nn.Linear(dim, k)
        self.linear_b = nn.Linear(dim, k)

    def forward(self, input_tensor_a, input_tensor_b,pertubation_attention=None):

        output_tensor_a = torch.relu(self.linear_a(input_tensor_a))
        output_tensor_b = torch.relu(self.linear_b(input_tensor_b))

        attention_matrix = torch.matmul(output_tensor_a, output_tensor_b.transpose(1, 2))  #[batchsize,1000,545]

        weights_proteins = torch.mean(attention_matrix, dim=1, keepdim=True)
        weights_proteins = F.softmax(weights_proteins,dim=-1)

        weights_compounds = torch.mean(attention_matrix, dim=2,keepdim=True) # [batchsize,1000,1]
        weights_compounds = F.softmax(weights_compounds,dim=1)

        attn_loss = 0
        if pertubation_attention is not None:
            attn_loss = stable_kl(weights_compounds,pertubation_attention)

        interactions_a = torch.matmul(weights_proteins, input_tensor_b)  # [batchsize,1,dim]
        interactions_b = torch.matmul(weights_compounds.transpose(1, 2), input_tensor_a)  # [batchsize,1,dim]

        return interactions_b.squeeze(1), interactions_a.squeeze(1), attention_matrix,attn_loss

'''Cross-Attention'''
class CrossAttention(nn.Module):
    def __init__(self, in_dim1, in_dim2, k_dim=64, v_dim=64, num_heads=2):
        super(CrossAttention, self).__init__()
        self.num_heads = num_heads
        self.k_dim = k_dim
        self.v_dim = v_dim

        self.proj_q1 = nn.Linear(in_dim1, k_dim * num_heads, bias=False)
        self.proj_k2 = nn.Linear(in_dim2, k_dim * num_heads, bias=False)
        self.proj_v2 = nn.Linear(in_dim2, v_dim * num_heads, bias=False)
        self.proj_oD = nn.Linear(v_dim * num_heads, in_dim1)


    def forward(self, x1, x2, mask=None):
        batch_size, seq_len1, in_dim1 = x1.size()
        seq_len2 = x2.size(1)

        q1 = self.proj_q1(x1).view(batch_size, seq_len1, self.num_heads, self.k_dim).permute(0, 2, 1, 3)
        k2 = self.proj_k2(x2).view(batch_size, seq_len2, self.num_heads, self.k_dim).permute(0, 2, 3, 1)

        v2 = self.proj_v2(x2).view(batch_size, seq_len2, self.num_heads, self.v_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q1, k2) / self.k_dim ** 0.5      #shape[batchsize, num_heads, seq_len1, seq_len2]
        # print(attn)
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)#.squeeze(1)
        # attn_atoms = torch.mean(attn,-1,keepdim=True).permute(0, 1, 3, 2)    #shape[batchsize, num_heads, 1, seq_len1]
        output = torch.matmul(attn, v2).permute(0, 2, 1, 3).contiguous().view(batch_size, seq_len1, -1)
        output = self.proj_oD(output)

        return output