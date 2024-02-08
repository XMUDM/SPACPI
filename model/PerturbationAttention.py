import torch
import torch.nn as nn
from torch.nn import functional as F
from utils.tools import parser
import warnings
warnings.filterwarnings("ignore")
args = parser()

class PerturbationAttention(nn.Module):
    def __init__(self, model):
        super(PerturbationAttention, self).__init__()
        self.model = model

    def lookup_emb(self, input):
        return self.model.DP_raw_fearure.drug_extractor(input)

    def generate_noise(self, embed, eps=1e-1):
        ''' Generate noise that conforms to a normal distribution with a variance of epsion
        and the same size as the compound embedded object'''
        noise = embed.data.new(embed.size()).normal_(0, 1) * eps
        noise.detach()
        noise.requires_grad_()
        return noise

    def cal_attn(self,delta,sparse_ratio=0.5):
        sigma = torch.norm(delta, dim=2, p=2).unsqueeze(-1)

        '''Calculate perturbation attention based on perturbation amplitude'''
        pertubation_attention = torch.tanh(1 - torch.div(sigma,torch.max(sigma)))
        pertubation_attention = F.log_softmax(pertubation_attention, 1).exp()

        '''Sparsify'''
        _, indices = torch.topk(-pertubation_attention, int(sparse_ratio*args.Cmax), dim=1)
        for i in range(pertubation_attention.size(0)):
            for j in range(int(sparse_ratio*args.Cmax)):
                pertubation_attention[i, indices[i, j], 0] = 0
        return pertubation_attention#.unsqueeze(-1)

